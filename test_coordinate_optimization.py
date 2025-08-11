"""
Test coordinate optimization for different optical systems.
"""

import torch
from config.system_config import SystemConfig
from simulation.builder import SystemBuilder

def test_coordinate_optimization():
    print("=" * 80)
    print("TESTING COORDINATE OPTIMIZATION")
    print("=" * 80)
    
    # Test different optical system configurations
    test_systems = [
        {
            'name': '550nm Short Focus System',
            'wavelength': 550e-9,
            'focal_length': 2e-3,  # 2mm focal length (short)
            'object_distance': 4e-3,  # 4mm object distance
            'aperture': 4e-3,  # 4mm aperture
        },
        {
            'name': '550nm Long Focus System', 
            'wavelength': 550e-9,
            'focal_length': 25e-3,  # 25mm focal length (long)
            'object_distance': 50e-3,  # 50mm object distance
            'aperture': 12e-3,  # 12mm aperture
        },
        {
            'name': '850nm IR System',
            'wavelength': 850e-9,  # Near-IR
            'focal_length': 10e-3,  # 10mm focal length
            'object_distance': 15e-3,  # 15mm object distance 
            'aperture': 8e-3,  # 8mm aperture
        }
    ]
    
    for i, system_params in enumerate(test_systems):
        print(f"\\n{i+1}. {system_params['name']}")
        print("-" * 60)
        
        # Create base config (will be optimized)
        config = SystemConfig(
            sensor_resolution=(64, 64),  # Start with small resolution
            wavelength=system_params['wavelength'],
            pixel_pitch=10e-6,  # Start with large pixel pitch
            propagation_distance=system_params['object_distance'] + system_params['focal_length']
        )
        
        print("Original config:")
        print(f"  Resolution: {config.sensor_resolution[0]}x{config.sensor_resolution[1]}")
        print(f"  Pixel pitch: {config.pixel_pitch*1e6:.1f}μm")
        print(f"  Field size: {config.sensor_resolution[0]*config.pixel_pitch*1e3:.1f}x{config.sensor_resolution[1]*config.pixel_pitch*1e3:.1f}mm")
        
        # Build system with coordinate optimization
        system = (SystemBuilder(config, auto_optimize_coordinates=True)
                  .add_lens(focal_length=system_params['focal_length'],
                           position=system_params['object_distance'],
                           spacing_after=system_params['focal_length'],
                           aperture_diameter=system_params['aperture'])
                  .build(auto_draw=False))  # Don't draw to keep output clean
        
        print("\\nAfter optimization:")
        print(f"  Resolution: {config.sensor_resolution[0]}x{config.sensor_resolution[1]}")
        print(f"  Pixel pitch: {config.pixel_pitch*1e6:.1f}μm")  
        print(f"  Field size: {config.sensor_resolution[0]*config.pixel_pitch*1e3:.1f}x{config.sensor_resolution[1]*config.pixel_pitch*1e3:.1f}mm")
        
        # Test the system with a simple field
        test_field = torch.complex(
            torch.ones(config.sensor_resolution, dtype=torch.float32) * 0.5,
            torch.zeros(config.sensor_resolution, dtype=torch.float32)
        )
        
        try:
            result = system(test_field)
            energy_conservation = result['intensity_sensor'].sum() / (torch.abs(test_field)**2).sum()
            print(f"  Energy conservation: {energy_conservation:.4f}")
            print(f"  Peak intensity: {result['intensity_sensor'].max():.3f}")
            print(f"  System working: YES")
        except Exception as e:
            print(f"  System working: NO - {str(e)}")
    
    print("\\n" + "=" * 80)
    print("COORDINATE OPTIMIZATION TEST COMPLETE")
    print("=" * 80)

def test_manual_vs_optimized():
    """Compare manual coordinate setup vs optimized."""
    print("\\n" + "=" * 80)
    print("MANUAL VS OPTIMIZED COORDINATE COMPARISON")
    print("=" * 80)
    
    # System parameters
    wavelength = 550e-9
    focal_length = 5e-3
    object_distance = 10e-3
    aperture = 8e-3
    
    # Manual setup (like current approach)
    print("\\n1. MANUAL COORDINATE SETUP")
    print("-" * 40)
    manual_config = SystemConfig(
        sensor_resolution=(128, 128),
        wavelength=wavelength,
        pixel_pitch=5e-6,
        propagation_distance=object_distance + focal_length
    )
    
    manual_system = (SystemBuilder(manual_config, auto_optimize_coordinates=False)
                    .add_lens(focal_length=focal_length,
                             position=object_distance,
                             spacing_after=focal_length,
                             aperture_diameter=aperture)
                    .build(auto_draw=False))
    
    # Optimized setup
    print("\\n2. OPTIMIZED COORDINATE SETUP")
    print("-" * 40)
    optimized_config = SystemConfig(
        sensor_resolution=(64, 64),  # Start smaller
        wavelength=wavelength,
        pixel_pitch=8e-6,  # Start larger
        propagation_distance=object_distance + focal_length
    )
    
    optimized_system = (SystemBuilder(optimized_config, auto_optimize_coordinates=True)
                       .add_lens(focal_length=focal_length,
                                position=object_distance,
                                spacing_after=focal_length,
                                aperture_diameter=aperture)
                       .build(auto_draw=False))
    
    # Compare performance
    print("\\n3. PERFORMANCE COMPARISON")
    print("-" * 40)
    
    # Create test field matching each system's resolution
    manual_field = torch.complex(
        torch.ones(manual_config.sensor_resolution, dtype=torch.float32) * 0.5,
        torch.zeros(manual_config.sensor_resolution, dtype=torch.float32)
    )
    
    optimized_field = torch.complex(
        torch.ones(optimized_config.sensor_resolution, dtype=torch.float32) * 0.5,
        torch.zeros(optimized_config.sensor_resolution, dtype=torch.float32)
    )
    
    # Test both systems
    try:
        manual_result = manual_system(manual_field)
        manual_conservation = manual_result['intensity_sensor'].sum() / (torch.abs(manual_field)**2).sum()
        print(f"Manual system energy conservation: {manual_conservation:.4f}")
    except Exception as e:
        print(f"Manual system failed: {e}")
    
    try:
        optimized_result = optimized_system(optimized_field) 
        optimized_conservation = optimized_result['intensity_sensor'].sum() / (torch.abs(optimized_field)**2).sum()
        print(f"Optimized system energy conservation: {optimized_conservation:.4f}")
    except Exception as e:
        print(f"Optimized system failed: {e}")

if __name__ == "__main__":
    test_coordinate_optimization()
    test_manual_vs_optimized()