"""
Clean RGB coherent imaging test using proper SystemBuilder.add_lens() method.
No more dummy phase masks!
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy import datasets
from scipy.ndimage import zoom
import os

# Import restructured components
from config.system_config import SystemConfig
from simulation.builder import SystemBuilder

def load_cameraman_image(target_size=(128, 128)):
    """Load and prepare cameraman test image."""
    try:
        cameraman = datasets.ascent()
        if cameraman.shape != target_size:
            zoom_factor = (target_size[0] / cameraman.shape[0], target_size[1] / cameraman.shape[1])
            cameraman = zoom(cameraman, zoom_factor, order=1)
        cameraman = cameraman.astype(np.float32) / 255.0
        print(f"Loaded real cameraman image: {cameraman.shape}")
    except:
        print("Creating simple test pattern...")
        y, x = np.meshgrid(np.linspace(-1, 1, target_size[0]), 
                          np.linspace(-1, 1, target_size[1]), indexing='ij')
        cameraman = np.zeros_like(x, dtype=np.float32)
        
        # Simple geometric pattern
        cameraman += 0.8 * ((x**2 + y**2) < 0.3).astype(float)  # Circle
        cameraman += 0.6 * ((np.abs(x) < 0.1) & (np.abs(y) < 0.6)).astype(float)  # Vertical bar
        cameraman += 0.6 * ((np.abs(y) < 0.1) & (np.abs(x) < 0.6)).astype(float)  # Horizontal bar
        cameraman = np.clip(cameraman, 0, 1)
    
    return torch.tensor(cameraman, dtype=torch.float32)

def test_clean_rgb_imaging():
    """Test RGB coherent imaging with clean SystemBuilder."""
    print("=" * 70)
    print("CLEAN RGB IMAGING TEST - NO DUMMY PHASE MASKS")
    print("=" * 70)
    
    # Load test image
    cameraman = load_cameraman_image((128, 128))
    input_field = torch.complex(torch.sqrt(cameraman.clamp(min=0)), 
                               torch.zeros_like(cameraman))
    
    # System parameters
    focal_length = 5e-3  # 5mm
    object_distance = 10e-3  # 10mm
    image_distance = 1 / (1/focal_length - 1/object_distance)  # 10mm
    aperture_diameter = 8e-3  # 8mm
    
    print(f"System: f={focal_length*1000:.1f}mm, object={object_distance*1000:.1f}mm")
    print(f"Image distance: {image_distance*1000:.1f}mm (theory)")
    
    # RGB wavelengths
    rgb_wavelengths = [640e-9, 550e-9, 450e-9]
    rgb_names = ['Red', 'Green', 'Blue'] 
    rgb_colors = ['red', 'green', 'blue']
    
    results = []
    
    for wavelength, name, color in zip(rgb_wavelengths, rgb_names, rgb_colors):
        print(f"\\n--- {name} ({wavelength*1e9:.0f}nm) ---")
        
        # Create system config
        config = SystemConfig(
            sensor_resolution=(128, 128),
            wavelength=wavelength,
            pixel_pitch=5e-6,
            propagation_distance=object_distance + image_distance
        )
        
        # Build system with PROPER lens method - no dummy elements!
        system = (SystemBuilder(config)
                  .add_lens(focal_length=focal_length,
                           position=object_distance,
                           spacing_after=image_distance,
                           aperture_diameter=aperture_diameter)
                  .build())
        
        print(f"  Built system with {len(system.elements)} elements")
        print(f"  Element type: {type(system.elements[0]).__name__}")
        
        # Run simulation
        output = system(input_field)
        
        # Extract results
        intensity = output['intensity_sensor'].squeeze().detach()
        field = output.get('field_sensor', None)
        if field is not None:
            field = field.squeeze().detach()
        
        # Diagnostics
        input_energy = torch.abs(input_field)**2
        input_energy_sum = input_energy.sum()
        energy_ratio = intensity.sum() / input_energy_sum
        
        print(f"  Output: mean={intensity.mean():.4f}, max={intensity.max():.4f}")
        print(f"  Energy conservation: {energy_ratio:.4f}")
        
        results.append({
            'name': name,
            'wavelength': wavelength,
            'intensity': intensity,
            'field': field,
            'color': color,
            'energy_ratio': energy_ratio.item()
        })
    
    # Create visualization
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    
    # Input
    axes[0, 0].imshow(cameraman.numpy(), cmap='gray')
    axes[0, 0].set_title('Input Image', fontweight='bold')
    axes[0, 0].axis('off')
    
    # RGB outputs
    for i, result in enumerate(results):
        axes[0, i+1].imshow(result['intensity'].numpy(), cmap='gray')
        axes[0, i+1].set_title(f'{result["name"]} Output\\n({result["wavelength"]*1e9:.0f}nm)', 
                              color=result['color'], fontweight='bold')
        axes[0, i+1].axis('off')
    
    # Cross-sections
    center_row = cameraman.shape[0] // 2
    axes[1, 0].plot(cameraman[center_row, :].numpy(), 'k-', linewidth=3, label='Input')
    for result in results:
        axes[1, 0].plot(result['intensity'][center_row, :].numpy(), 
                       color=result['color'], linewidth=2, label=f'{result["name"]}')
    axes[1, 0].set_title('Cross-sections')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Energy conservation
    energy_ratios = [r['energy_ratio'] for r in results]
    axes[1, 1].bar(range(3), energy_ratios, color=[r['color'] for r in results], alpha=0.7)
    axes[1, 1].set_title('Energy Conservation')
    axes[1, 1].set_xticks(range(3))
    axes[1, 1].set_xticklabels([r['name'] for r in results])
    axes[1, 1].axhline(y=1.0, color='black', linestyle='--', alpha=0.5)
    axes[1, 1].grid(True, alpha=0.3)
    
    # Statistics
    axes[1, 2].axis('off')
    stats_text = "Clean Builder Results:\\n\\n"
    stats_text += f"+ No dummy phase masks\\n"
    stats_text += f"+ Direct lens elements\\n"
    stats_text += f"+ Proper SystemBuilder.add_lens()\\n\\n"
    
    for result in results:
        stats_text += f"{result['name']}: Energy={result['energy_ratio']:.3f}\\n"
    
    axes[1, 2].text(0.05, 0.95, stats_text, transform=axes[1, 2].transAxes,
                   fontsize=11, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    
    # Comparison
    axes[1, 3].axis('off')
    comparison_text = f"""OLD Way (BAD):
system = builder.add_phase_mask(...)
system.elements[0] = lens  # Replace!

NEW Way (GOOD):  
system = builder.add_lens(
    focal_length=5e-3,
    aperture_diameter=8e-3
).build()

Much cleaner!"""
    
    axes[1, 3].text(0.05, 0.95, comparison_text, transform=axes[1, 3].transAxes,
                   fontsize=10, verticalalignment='top', fontfamily='monospace',
                   bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    plt.suptitle('Clean RGB Imaging - Proper SystemBuilder.add_lens()', 
                fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    os.makedirs('results', exist_ok=True)
    plt.savefig('results/rgb_clean_builder.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    return results

if __name__ == "__main__":
    results = test_clean_rgb_imaging()
    
    print("\\n" + "=" * 70)
    print("CLEAN BUILDER TEST COMPLETE")
    print("=" * 70)
    print("Key improvements:")
    print("+ No more dummy phase masks")
    print("+ Direct lens creation with SystemBuilder.add_lens()")
    print("+ Cleaner, more intuitive API")
    print("+ Proper element positioning and spacing")
    print("\\nNow you can build lens systems properly:")
    print("  system = SystemBuilder(config).add_lens(focal_length=5e-3).build()")
    print("=" * 70)