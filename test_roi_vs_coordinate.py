"""
Test and compare ROI-based system vs coordinate optimization approach.

This demonstrates the difference between:
1. Traditional: Static field sizing with no optimization
2. New ROI: Dynamic ROI tracking + aperture-based resets
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from config.system_config import SystemConfig
from simulation.builder import SystemBuilder
from utils.test_images import create_test_image, ImageType

def test_roi_vs_coordinate_optimization():
    """Compare ROI-based system with traditional approach."""
    print("=" * 80)
    print("ROI-BASED vs TRADITIONAL APPROACH COMPARISON")
    print("=" * 80)
    
    # System parameters
    design_wavelength = 550e-9
    focal_length = 25e-3
    object_distance = 50e-3
    aperture_diameter = 3e-3  # 3mm aperture
    
    # Create identical system configurations
    config_traditional = SystemConfig(
        sensor_resolution=(256, 256),
        wavelength=design_wavelength,
        pixel_pitch=4e-6,
        propagation_distance=0.0
    )
    
    config_roi = SystemConfig(
        sensor_resolution=(256, 256),
        wavelength=design_wavelength,
        pixel_pitch=4e-6,
        propagation_distance=0.0
    )
    
    print(f"Base system parameters:")
    print(f"  Wavelength: {design_wavelength*1e9:.0f}nm")
    print(f"  Lens: {focal_length*1e3:.1f}mm focal length")
    print(f"  Aperture: {aperture_diameter*1e3:.1f}mm diameter")
    print(f"  Initial resolution: {config_traditional.sensor_resolution}")
    print(f"  Initial pixel pitch: {config_traditional.pixel_pitch*1e6:.1f}μm")
    print()
    
    # Test 1: Traditional approach (no coordinate optimization)
    print("TEST 1: TRADITIONAL APPROACH (STATIC FIELD SIZING)")
    print("-" * 50)
    
    system_traditional = (SystemBuilder(config_traditional, auto_optimize_coordinates=False, use_roi_system=False)
                         .add_lens(focal_length=focal_length,
                                  position=object_distance,
                                  spacing_after=abs(1 / (1/focal_length - 1/object_distance)),
                                  aperture_diameter=aperture_diameter)
                         .build(auto_draw=False))
    
    print(f"Final traditional config:")
    print(f"  Resolution: {config_traditional.sensor_resolution}")
    print(f"  Pixel pitch: {config_traditional.pixel_pitch*1e6:.1f}μm")
    print(f"  Field size: {config_traditional.sensor_resolution[0]*config_traditional.pixel_pitch*1e3:.2f}mm")
    print()
    
    # Test 2: ROI-based approach  
    print("TEST 2: ROI-BASED DYNAMIC APPROACH")
    print("-" * 50)
    
    system_roi = (SystemBuilder(config_roi, auto_optimize_coordinates=False, use_roi_system=True)
                 .add_lens(focal_length=focal_length,
                          position=object_distance,
                          spacing_after=abs(1 / (1/focal_length - 1/object_distance)),
                          aperture_diameter=aperture_diameter)
                 .build(auto_draw=False))
    
    # Test with text image
    print("TESTING WITH TEXT IMAGE")
    print("-" * 30)
    
    text_image = create_test_image((256, 256), ImageType.TEXT)
    text_field = torch.complex(text_image, torch.zeros_like(text_image))
    
    # Enable caching for both systems
    system_traditional.set_cache_intermediates(True)
    system_roi.set_cache_intermediates(True)
    
    # Run both systems
    print("\\nRunning traditional system...")
    result_traditional = system_traditional(text_field)
    
    print("\\nRunning ROI-based system...")
    result_roi = system_roi(text_field)
    
    # Print ROI evolution
    if hasattr(system_roi, 'print_roi_evolution'):
        system_roi.print_roi_evolution()
    
    # Compare results
    print("\\n" + "="*60)
    print("RESULTS COMPARISON")
    print("="*60)
    
    energy_traditional = result_traditional['intensity_sensor'].sum().item()
    energy_roi = result_roi['intensity_sensor'].sum().item()
    
    print(f"Traditional approach:")
    print(f"  Output energy: {energy_traditional:.1f}")
    print(f"  Peak intensity: {result_traditional['intensity_sensor'].max():.3f}")
    print(f"  Final resolution: {result_traditional['intensity_sensor'].shape}")
    
    print(f"\\nROI-based system:")
    print(f"  Output energy: {energy_roi:.1f}")
    print(f"  Peak intensity: {result_roi['intensity_sensor'].max():.3f}")
    print(f"  Final resolution: {result_roi['intensity_sensor'].shape}")
    
    print(f"\\nComparison:")
    print(f"  Energy ratio (ROI/Traditional): {energy_roi/energy_traditional:.3f}")
    print(f"  Peak ratio (ROI/Traditional): {result_roi['intensity_sensor'].max()/result_traditional['intensity_sensor'].max():.3f}")
    
    # Create comparison visualization
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    
    # Input
    axes[0, 0].imshow(text_image.numpy(), cmap='gray')
    axes[0, 0].set_title('Input: Text')
    axes[0, 0].axis('off')
    axes[1, 0].axis('off')  # Empty
    
    # Traditional results
    traditional_sensor = result_traditional['intensity_sensor'].detach().numpy()
    axes[0, 1].imshow(traditional_sensor, cmap='hot')
    axes[0, 1].set_title('Traditional: Sensor')
    axes[0, 1].axis('off')
    
    # ROI system results
    roi_sensor = result_roi['intensity_sensor'].detach().numpy()
    axes[0, 2].imshow(roi_sensor, cmap='hot')
    axes[0, 2].set_title('ROI System: Sensor')
    axes[0, 2].axis('off')
    
    # Difference
    if traditional_sensor.shape == roi_sensor.shape:
        diff = np.abs(traditional_sensor - roi_sensor)
        axes[0, 3].imshow(diff, cmap='viridis')
        axes[0, 3].set_title('|Difference|')
        axes[0, 3].axis('off')
    else:
        axes[0, 3].text(0.5, 0.5, f'Different\\nResolutions\\nTraditional: {traditional_sensor.shape}\\nROI: {roi_sensor.shape}',
                       ha='center', va='center', transform=axes[0, 3].transAxes)
        axes[0, 3].axis('off')
    
    # Cross-section profiles
    center_row = traditional_sensor.shape[0] // 2
    traditional_profile = traditional_sensor[center_row, :]
    
    if roi_sensor.shape[0] == traditional_sensor.shape[0]:
        roi_profile = roi_sensor[center_row, :]
        axes[1, 1].plot(traditional_profile, 'b-', label='Traditional', linewidth=2)
        axes[1, 1].plot(roi_profile, 'r--', label='ROI System', linewidth=2, alpha=0.8)
        axes[1, 1].set_title('Cross-section Comparison')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    else:
        axes[1, 1].plot(traditional_profile, 'b-', label='Traditional', linewidth=2)
        axes[1, 1].set_title('Traditional Profile')
        axes[1, 1].grid(True, alpha=0.3)
        
        roi_center = roi_sensor.shape[0] // 2
        roi_profile = roi_sensor[roi_center, :]
        axes[1, 2].plot(roi_profile, 'r-', label='ROI System', linewidth=2)
        axes[1, 2].set_title('ROI System Profile')
        axes[1, 2].grid(True, alpha=0.3)
    
    # Energy comparison
    energies = [energy_traditional, energy_roi]
    methods = ['Traditional\\nApproach', 'ROI-based\\nSystem']
    axes[1, 3].bar(methods, energies, color=['blue', 'red'], alpha=0.7)
    axes[1, 3].set_title('Output Energy Comparison')
    axes[1, 3].set_ylabel('Energy')
    
    plt.suptitle('ROI-based vs Traditional Approach Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('results/roi_vs_traditional_comparison.png', dpi=150)
    print(f"\\nSaved comparison to: results/roi_vs_traditional_comparison.png")
    
    return system_traditional, system_roi, result_traditional, result_roi

if __name__ == "__main__":
    traditional_sys, roi_sys, traditional_result, roi_result = test_roi_vs_coordinate_optimization()