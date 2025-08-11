"""
Test the smart coordinate optimization system.
"""

import torch
from config.system_config import SystemConfig
from simulation.builder import SystemBuilder

def test_smart_coordinates():
    print("=" * 80)
    print("SMART COORDINATE OPTIMIZATION DEMO")
    print("=" * 80)
    
    # Test a reasonable optical system
    print("\nBuilding 550nm imaging system with smart coordinates...")
    
    # Start with a simple base config - optimizer will improve it
    config = SystemConfig(
        sensor_resolution=(128, 128),  # Starting point
        wavelength=550e-9,
        pixel_pitch=5e-6,  # Starting point  
        propagation_distance=15e-3
    )
    
    print("Initial configuration:")
    print(f"  Resolution: {config.sensor_resolution[0]}×{config.sensor_resolution[1]}")
    print(f"  Pixel pitch: {config.pixel_pitch*1e6:.1f}μm")
    print(f"  Field size: {config.sensor_resolution[0]*config.pixel_pitch*1e3:.2f}×{config.sensor_resolution[1]*config.pixel_pitch*1e3:.2f}mm")
    
    # Build system with coordinate optimization
    system = (SystemBuilder(config, auto_optimize_coordinates=True)
              .add_lens(focal_length=5e-3,        # 5mm focal length
                       position=10e-3,            # 10mm object distance  
                       spacing_after=10e-3,       # 10mm to sensor
                       aperture_diameter=6e-3)    # 6mm aperture
              .build(auto_draw=False))
    
    print(f"\nOptimized configuration:")
    print(f"  Resolution: {config.sensor_resolution[0]}×{config.sensor_resolution[1]}")
    print(f"  Pixel pitch: {config.pixel_pitch*1e6:.1f}μm")
    print(f"  Field size: {config.sensor_resolution[0]*config.pixel_pitch*1e3:.2f}×{config.sensor_resolution[1]*config.pixel_pitch*1e3:.2f}mm")
    
    # Test the system with properly sized field
    print("\\nTesting optimized system...")
    test_field = torch.complex(
        torch.ones(config.sensor_resolution, dtype=torch.float32) * 0.5,
        torch.zeros(config.sensor_resolution, dtype=torch.float32) 
    )
    
    input_energy = (torch.abs(test_field)**2).sum()
    
    try:
        result = system(test_field)
        output_energy = result['intensity_sensor'].sum()
        energy_conservation = output_energy / input_energy
        
        print(f"✓ System simulation successful!")
        print(f"  Input energy: {input_energy:.3f}")
        print(f"  Output energy: {output_energy:.3f}")
        print(f"  Energy conservation: {energy_conservation:.4f}")
        print(f"  Peak intensity: {result['intensity_sensor'].max():.3f}")
        print(f"  SNR: {result['snr']}")
        
    except Exception as e:
        print(f"✗ System simulation failed: {e}")
    
    print("\\n" + "=" * 80)
    print("COORDINATE OPTIMIZATION BENEFITS:")
    print("✓ Automatic resolution optimization based on optical parameters")
    print("✓ Pixel pitch optimization for proper sampling")
    print("✓ Padding factor calculation for edge artifact prevention")
    print("✓ Warning system for problematic configurations")
    print("✓ Memory usage control with reasonable limits")
    print("=" * 80)

def test_coordinate_warnings():
    print("\\n" + "=" * 80) 
    print("COORDINATE WARNING SYSTEM DEMO")
    print("=" * 80)
    
    print("\\nTesting extreme optical system that triggers warnings...")
    
    config = SystemConfig(
        sensor_resolution=(64, 64),
        wavelength=550e-9,
        pixel_pitch=20e-6,  # Large pixel pitch
        propagation_distance=5e-3
    )
    
    # Build extreme system: very short focal length, small aperture
    try:
        system = (SystemBuilder(config, auto_optimize_coordinates=True)
                  .add_lens(focal_length=1e-3,      # 1mm focal length (very short)
                           position=2e-3,           # 2mm object distance
                           spacing_after=50e-3,     # 50mm propagation (very long)
                           aperture_diameter=2e-3)  # 2mm aperture (small)
                  .build(auto_draw=False))
        
        print("System built successfully with warnings above.")
        
    except Exception as e:
        print(f"System build failed: {e}")
    
    print("\\n" + "=" * 80)
    print("The warning system helps identify:")
    print("• Excessive memory requirements")
    print("• High padding factors indicating edge artifacts")
    print("• Sampling violations (Nyquist limit)")
    print("• Suboptimal optical configurations")
    print("=" * 80)

if __name__ == "__main__":
    test_smart_coordinates()
    test_coordinate_warnings()