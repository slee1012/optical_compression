"""
Focused RGB coherent imaging test to diagnose the imaging issues.
Shows input image intensity/phase and separate R/G/B coherent simulations.
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
from physics.light_sources import SpectralSource, SpectrumType
from elements.lens import create_thin_lens

def load_cameraman_image(target_size=(128, 128)):
    """Load and prepare cameraman test image."""
    try:
        # Try to load cameraman from scipy
        cameraman = datasets.ascent()
        
        # Resize to target size
        if cameraman.shape != target_size:
            zoom_factor = (target_size[0] / cameraman.shape[0], target_size[1] / cameraman.shape[1])
            cameraman = zoom(cameraman, zoom_factor, order=1)
        
        # Normalize to [0, 1]
        cameraman = cameraman.astype(np.float32) / 255.0
        
        print(f"Loaded real cameraman image: {cameraman.shape}")
        
    except:
        # Create a clear, simple test pattern if cameraman fails
        print("Creating simple test pattern...")
        y, x = np.meshgrid(np.linspace(-1, 1, target_size[0]), 
                          np.linspace(-1, 1, target_size[1]), indexing='ij')
        
        # Simple geometric pattern - should look like clear shapes after imaging
        cameraman = np.zeros_like(x, dtype=np.float32)
        
        # Large central circle
        cameraman += 0.8 * ((x**2 + y**2) < 0.3).astype(float)
        
        # Rectangular bars
        cameraman += 0.6 * ((np.abs(x) < 0.1) & (np.abs(y) < 0.6)).astype(float)
        cameraman += 0.6 * ((np.abs(y) < 0.1) & (np.abs(x) < 0.6)).astype(float)
        
        # Small corner squares
        cameraman += 0.4 * ((np.abs(x - 0.6) < 0.1) & (np.abs(y - 0.6) < 0.1)).astype(float)
        cameraman += 0.4 * ((np.abs(x + 0.6) < 0.1) & (np.abs(y + 0.6) < 0.1)).astype(float)
        
        # Normalize
        cameraman = np.clip(cameraman, 0, 1)
    
    return torch.tensor(cameraman, dtype=torch.float32)

def test_rgb_coherent_imaging():
    """Test coherent imaging separately for R, G, B wavelengths."""
    print("=" * 70)
    print("RGB COHERENT IMAGING DIAGNOSTIC TEST")
    print("=" * 70)
    
    # Load test image
    cameraman = load_cameraman_image((128, 128))
    print(f"Input image shape: {cameraman.shape}")
    print(f"Input range: [{cameraman.min():.3f}, {cameraman.max():.3f}]")
    
    # Convert to complex field (intensity to amplitude)
    input_field = torch.complex(torch.sqrt(cameraman.clamp(min=0)), 
                               torch.zeros_like(cameraman))
    
    print(f"Input field amplitude range: [{torch.abs(input_field).min():.3f}, {torch.abs(input_field).max():.3f}]")
    print(f"Input field phase range: [{torch.angle(input_field).min():.3f}, {torch.angle(input_field).max():.3f}] rad")
    
    # System parameters - proper imaging system
    focal_length = 5e-3  # 5mm
    object_distance = 10e-3  # 10mm (object to lens)
    
    # Calculate image distance using lens equation: 1/f = 1/o + 1/i
    # For f=5mm, o=10mm: i = 1/(1/5 - 1/10) = 10mm
    image_distance = 1 / (1/focal_length - 1/object_distance) 
    print(f"\\nLens equation: f={focal_length*1000:.1f}mm, object={object_distance*1000:.1f}mm")
    print(f"Calculated image distance: {image_distance*1000:.1f}mm from lens")
    print(f"Total propagation distance: {(object_distance + image_distance)*1000:.1f}mm")
    
    aperture_diameter = 8e-3  # 8mm
    
    # RGB wavelengths
    rgb_wavelengths = [640e-9, 550e-9, 450e-9]  # Red, Green, Blue
    rgb_names = ['Red', 'Green', 'Blue']
    rgb_colors = ['red', 'green', 'blue']
    
    # Test each wavelength separately
    rgb_results = []
    
    for i, (wavelength, name, color) in enumerate(zip(rgb_wavelengths, rgb_names, rgb_colors)):
        print(f"\\n--- Testing {name} ({wavelength*1e9:.0f}nm) ---")
        
        # Create system configuration for this wavelength
        config = SystemConfig(
            sensor_resolution=(128, 128),
            wavelength=wavelength,
            pixel_pitch=5e-6,  # 5μm pixels
            propagation_distance=object_distance + image_distance
        )
        
        # Create lens element for this wavelength
        lens = create_thin_lens(
            focal_length=focal_length,
            aperture_diameter=aperture_diameter,
            resolution=config.sensor_resolution,
            pixel_pitch=config.pixel_pitch,
            wavelength=wavelength
        )
        
        # Build system
        system = (SystemBuilder(config)
                  .add_phase_mask(position=object_distance, spacing_after=image_distance,
                                clear_aperture=aperture_diameter)
                  .build())
        
        # Replace phase mask with proper lens
        system.elements[0] = lens
        
        print(f"  System wavelength: {wavelength*1e9:.0f}nm")
        print(f"  Propagation distance: {config.propagation_distance*1000:.1f}mm")
        print(f"  Lens focal length: {focal_length*1000:.1f}mm")
        
        # Run simulation
        output = system(input_field)
        
        # Extract results
        if isinstance(output, dict):
            if 'intensity_sensor' in output:
                intensity = output['intensity_sensor']
                field = output.get('field_sensor', None)
            else:
                intensity = torch.abs(output['field_sensor'])**2 if 'field_sensor' in output else None
                field = output.get('field_sensor', None)
        else:
            intensity = torch.abs(output)**2
            field = output
        
        # Remove batch dimensions if present
        if intensity.ndim == 3:
            intensity = intensity.squeeze(0)
        if field is not None and field.ndim == 3:
            field = field.squeeze(0)
        
        rgb_results.append({
            'name': name,
            'wavelength': wavelength,
            'intensity': intensity.detach(),
            'field': field.detach() if field is not None else None,
            'color': color
        })
        
        # Print diagnostics
        input_energy_tensor = torch.abs(input_field)**2
        input_energy = input_energy_tensor.sum()
        output_energy = intensity.sum()
        energy_ratio = output_energy / input_energy
        
        print(f"  Output intensity: mean={intensity.mean():.4f}, max={intensity.max():.4f}")
        print(f"  Energy conservation: {energy_ratio:.4f}")
        print(f"  Non-zero pixels: {(intensity > 1e-6).sum()}/{intensity.numel()}")
    
    # Create comprehensive visualization
    print("\\nCreating visualization...")
    fig = plt.figure(figsize=(20, 12))
    
    # Layout: 4 rows x 5 columns
    # Row 0: Input intensity, Input phase, empty, empty, empty  
    # Row 1: Red intensity, Green intensity, Blue intensity, RGB comparison, Energy plot
    # Row 2: Red phase, Green phase, Blue phase, Cross-sections, Statistics
    # Row 3: Red field amplitude, Green field amplitude, Blue field amplitude, Phase comparison, empty
    
    # Input image intensity
    ax = plt.subplot(4, 5, 1)
    plt.imshow(cameraman.numpy(), cmap='gray')
    plt.title('Input Image\\n(Intensity)', fontsize=12, weight='bold')
    plt.axis('off')
    plt.colorbar(fraction=0.046)
    
    # Input field phase
    ax = plt.subplot(4, 5, 2)
    input_phase = torch.angle(input_field).numpy()
    plt.imshow(input_phase, cmap='twilight', vmin=-np.pi, vmax=np.pi)
    plt.title('Input Field\\n(Phase)', fontsize=12, weight='bold')
    plt.axis('off')
    plt.colorbar(fraction=0.046)
    
    # RGB intensity outputs
    for i, result in enumerate(rgb_results):
        ax = plt.subplot(4, 5, 6 + i)  # Row 1
        plt.imshow(result['intensity'].numpy(), cmap='gray')
        plt.title(f'{result["name"]} Output\\n({result["wavelength"]*1e9:.0f}nm)', 
                 fontsize=11, color=result['color'], weight='bold')
        plt.axis('off')
        
        # RGB phases
        if result['field'] is not None:
            ax = plt.subplot(4, 5, 11 + i)  # Row 2  
            phase = torch.angle(result['field']).numpy()
            plt.imshow(phase, cmap='twilight', vmin=-np.pi, vmax=np.pi)
            plt.title(f'{result["name"]} Phase', fontsize=10, color=result['color'])
            plt.axis('off')
            
            # Field amplitudes
            ax = plt.subplot(4, 5, 16 + i)  # Row 3
            amplitude = torch.abs(result['field']).numpy()
            plt.imshow(amplitude, cmap='gray')
            plt.title(f'{result["name"]} Amplitude', fontsize=10, color=result['color'])
            plt.axis('off')
    
    # RGB comparison
    ax = plt.subplot(4, 5, 9)
    rgb_means = [r['intensity'].mean().item() for r in rgb_results]
    rgb_maxes = [r['intensity'].max().item() for r in rgb_results]
    
    x_pos = np.arange(3)
    width = 0.35
    
    plt.bar(x_pos - width/2, rgb_means, width, label='Mean', alpha=0.7, 
           color=[r['color'] for r in rgb_results])
    plt.bar(x_pos + width/2, rgb_maxes, width, label='Max', alpha=0.7,
           color=[r['color'] for r in rgb_results])
    
    plt.title('RGB Intensity Comparison', fontsize=11, weight='bold')
    plt.xticks(x_pos, [r['name'] for r in rgb_results])
    plt.ylabel('Intensity')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Energy conservation
    ax = plt.subplot(4, 5, 10)
    input_energy_tensor = torch.abs(input_field)**2
    input_energy = input_energy_tensor.sum()
    energy_ratios = [r['intensity'].sum() / input_energy for r in rgb_results]
    
    plt.bar(range(3), energy_ratios, color=[r['color'] for r in rgb_results], alpha=0.7)
    plt.title('Energy Conservation', fontsize=11, weight='bold')
    plt.xticks(range(3), [r['name'] for r in rgb_results])
    plt.ylabel('Output/Input Energy')
    plt.axhline(y=1.0, color='black', linestyle='--', alpha=0.5)
    plt.grid(True, alpha=0.3)
    
    # Cross-sections
    ax = plt.subplot(4, 5, 14)
    center_row = cameraman.shape[0] // 2
    
    # Input
    plt.plot(cameraman[center_row, :].numpy(), 'k-', linewidth=3, alpha=0.7, label='Input')
    
    # RGB outputs
    for result in rgb_results:
        plt.plot(result['intensity'][center_row, :].numpy(), 
                color=result['color'], linewidth=2, label=f'{result["name"]} Output')
    
    plt.title('Center Row Cross-sections', fontsize=11, weight='bold')
    plt.xlabel('Pixel')
    plt.ylabel('Intensity')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Statistics table
    ax = plt.subplot(4, 5, 15)
    ax.axis('off')
    
    stats_text = "RGB Statistics:\\n\\n"
    stats_text += f"{'Channel':<8} {'Mean':<8} {'Max':<8} {'Energy':<8}\\n"
    stats_text += "-" * 35 + "\\n"
    
    for result in rgb_results:
        energy_ratio = result['intensity'].sum() / input_energy
        stats_text += f"{result['name']:<8} {result['intensity'].mean():.4f} {result['intensity'].max():.4f} {energy_ratio:.4f}\\n"
    
    stats_text += f"\\nInput energy: {input_energy:.2f}\\n"
    stats_text += f"Expected: Sharp image features\\n"
    stats_text += f"Actual: {'Sharp' if rgb_maxes[1] > 0.5 else 'Diffracted'} patterns"
    
    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=10,
           verticalalignment='top', fontfamily='monospace',
           bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    plt.suptitle('RGB Coherent Imaging Diagnostic - Separate Wavelengths', 
                fontsize=16, weight='bold')
    plt.tight_layout()
    
    # Save results
    os.makedirs('results', exist_ok=True)
    plt.savefig('results/rgb_coherent_diagnostic.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    return {
        'input': {'intensity': cameraman, 'field': input_field},
        'rgb_results': rgb_results
    }

if __name__ == "__main__":
    results = test_rgb_coherent_imaging()
    
    print("\\n" + "=" * 70)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 70)
    print("Generated: results/rgb_coherent_diagnostic.png")
    print("\\nThis shows:")
    print("1. Input image intensity and phase")
    print("2. Separate R/G/B coherent imaging results")
    print("3. Energy conservation analysis")
    print("4. Cross-section comparisons")
    print("\\nUse this to diagnose why the imaging looks like diffraction patterns")
    print("instead of properly focused images.")
    print("=" * 70)