"""
Build a simple imaging system with 550nm design wavelength.

Usage:
  python build_550nm_system.py                    # Default behavior
  python build_550nm_system.py display=True       # Show plots on screen
  python build_550nm_system.py verbose=True       # Enable verbose output
  python build_550nm_system.py save=False         # Disable saving
"""

import torch
from config.system_config import SystemConfig
from simulation.builder import SystemBuilder
from utils.cli_args import OpticalScriptManager
from utils.test_images import create_test_image, ImageType

def build_550nm_imaging_system():
    # Setup CLI argument handling
    script_manager = OpticalScriptManager("Build 550nm System")
    script_manager.print_header("BUILDING 550NM IMAGING SYSTEM")
    
    # System parameters for 550nm green light
    design_wavelength = 550e-9  # 550nm green light
    focal_length = 25e-3         # 5mm focal length
    object_distance = 50e-3     # 10mm from object to lens
    aperture_diameter = 4e-3    # 8mm aperture
    
    # Create system configuration
    config = SystemConfig(
        sensor_resolution=(512, 512),     # 128x128 pixels
        wavelength=design_wavelength,     # Design wavelength
        pixel_pitch=2e-6,                # 5μm pixels
        propagation_distance=0.0         # Use explicit element positioning instead
    )
    
    script_manager.print_unless_quiet(f"Design wavelength: {design_wavelength*1e9:.0f}nm")
    script_manager.print_unless_quiet(f"Lens focal length: {focal_length*1000:.1f}mm")
    script_manager.print_unless_quiet(f"Object distance: {object_distance*1000:.1f}mm")
    script_manager.print_unless_quiet(f"Aperture diameter: {aperture_diameter*1000:.1f}mm")
    script_manager.print_unless_quiet(f"Sensor: {config.sensor_resolution[0]}x{config.sensor_resolution[1]} pixels")
    script_manager.print_unless_quiet(f"Pixel pitch: {config.pixel_pitch*1e6:.1f}μm")
    
    # Calculate theoretical image distance using lens equation
    # 1/f = 1/s_o + 1/s_i  ->  s_i = 1/(1/f - 1/s_o)
    s_i = 1 / (1/focal_length - 1/object_distance)
    theoretical_image_distance = object_distance + abs(s_i)
    
    script_manager.print_unless_quiet(f"\\nTheoretical image distance: {theoretical_image_distance*1000:.1f}mm")
    script_manager.print_unless_quiet(f"Magnification: {abs(s_i/object_distance):.3f}x")
    
    # Build the optical system
    script_manager.print_section("Building optical system")
    
    # Determine coordinate optimization preference
    optimize_coords = script_manager.should_optimize_coords()
    if optimize_coords is None:
        optimize_coords = False  # Disable until element rebuilding is fixed
    
    # Setup visualization with CLI control
    from analysis.visualization import Visualizer
    visualizer = script_manager.setup_visualization()
    
    system = (SystemBuilder(config, auto_optimize_coordinates=optimize_coords)
              .add_lens(focal_length=focal_length,
                       position=object_distance,           # Lens at 10mm
                       spacing_after=abs(s_i),            # Distance to image
                       aperture_diameter=aperture_diameter)
              .build(auto_draw=script_manager.should_save()))  # Auto-draw based on save preference
    
    script_manager.print_unless_quiet(f"System built with {len(system.elements)} optical element(s)")
    if script_manager.should_save():
        script_manager.record_result("results/optical_system_*.png")
    
    # Test with various input images
    script_manager.print_section("Testing system with test images")
    
    # Parse image type from CLI arguments (default to gaussian)
    image_type_str = script_manager.args.get('image', 'gaussian')
    try:
        image_type = ImageType(image_type_str.lower())
    except ValueError:
        script_manager.print_unless_quiet(f"Unknown image type '{image_type_str}', using gaussian")
        image_type = ImageType.GAUSSIAN
    
    # IMPORTANT: Create test image at the ORIGINAL input resolution (before optimization)
    original_resolution = (1024, 1024)  # The resolution user specified
    script_manager.print_unless_quiet(f"Creating {image_type.value} test image at {original_resolution[0]}×{original_resolution[1]}...")
    test_image = create_test_image(original_resolution, image_type)
    
    # If coordinate optimization changed the resolution, we need to pad the input image
    if config.sensor_resolution != original_resolution:
        script_manager.print_unless_quiet(f"Padding input image from {original_resolution} to {config.sensor_resolution}")
        
        # Calculate padding needed
        pad_h = (config.sensor_resolution[0] - original_resolution[0]) // 2
        pad_w = (config.sensor_resolution[1] - original_resolution[1]) // 2
        
        # Pad the image symmetrically
        import torch.nn.functional as F
        test_image = F.pad(test_image, (pad_w, pad_w, pad_h, pad_h), mode='constant', value=0)
    
    # Convert to complex field (amplitude with zero phase)
    test_field = torch.complex(test_image, torch.zeros_like(test_image))
    
    input_energy = (torch.abs(test_field)**2).sum()
    
    # Enable intermediate field caching to see field profiles
    system.set_cache_intermediates(True)
    
    # Propagate through the system
    result = system(test_field)
    
    output_energy = result['intensity_sensor'].sum()
    energy_ratio = output_energy / input_energy
    
    script_manager.print_unless_quiet(f"Input energy: {input_energy:.3f}")
    script_manager.print_unless_quiet(f"Output energy: {output_energy:.3f}")
    script_manager.print_unless_quiet(f"Energy conservation: {energy_ratio:.3f}")
    script_manager.print_unless_quiet(f"Peak intensity: {result['intensity_sensor'].max():.3f}")
    script_manager.print_unless_quiet(f"SNR: {result['snr']}")
    
    # Display intermediate field information
    script_manager.print_section("Intermediate field analysis")
    system.list_intermediate_fields()
    
    # Extract key intermediate fields for visualization
    before_lens_field = system.get_intermediate_field(label_contains="Before ThinLens")
    after_lens_field = system.get_intermediate_field(label_contains="After ThinLens")
    
    # Create and save simulation results visualization
    script_manager.print_section("Creating simulation visualization")
    
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Create comprehensive visualization showing amplitude and phase before/after lens
    if before_lens_field and after_lens_field:
        # Full analysis with before/after lens amplitude and phase
        fig = plt.figure(figsize=(20, 12))
        
        # Create subplot grid: 3 rows x 4 columns
        # Row 1: 2D images (Input, Before Lens Intensity, After Lens Intensity, Sensor Output)
        # Row 2: Before/After Lens Amplitude profiles
        # Row 3: Before/After Lens Phase profiles
        
        # Row 1: 2D intensity images
        ax1 = plt.subplot(3, 4, 1)
        ax1.imshow(test_image.numpy(), cmap='gray')
        ax1.set_title(f'Input: {image_type.value.title()}')
        ax1.axis('off')
        
        ax2 = plt.subplot(3, 4, 2)
        before_lens_intensity = (torch.abs(before_lens_field['field'])**2).detach().numpy()
        ax2.imshow(before_lens_intensity, cmap='viridis')
        ax2.set_title(f'Before Lens\n({before_lens_field["position"]*1e3:.1f}mm)')
        ax2.axis('off')
        
        ax3 = plt.subplot(3, 4, 3)
        after_lens_intensity = (torch.abs(after_lens_field['field'])**2).detach().numpy()
        ax3.imshow(after_lens_intensity, cmap='viridis')
        ax3.set_title(f'After Lens\n({after_lens_field["position"]*1e3:.1f}mm)')
        ax3.axis('off')
        
        ax4 = plt.subplot(3, 4, 4)
        sensor_intensity = result['intensity_sensor'].detach().numpy()
        ax4.imshow(sensor_intensity, cmap='hot')
        ax4.set_title('Sensor Output')
        ax4.axis('off')
        
        # Get center row cross-sections
        center_row = sensor_intensity.shape[0] // 2
        x_pixels = np.arange(sensor_intensity.shape[1])
        
        # Extract complex field data
        before_lens_complex = before_lens_field['field'][center_row, :].detach()
        after_lens_complex = after_lens_field['field'][center_row, :].detach()
        
        before_amplitude = torch.abs(before_lens_complex).numpy()
        after_amplitude = torch.abs(after_lens_complex).numpy()
        before_phase = torch.angle(before_lens_complex).numpy()
        after_phase = torch.angle(after_lens_complex).numpy()
        
        # Row 2: Amplitude profiles
        ax5 = plt.subplot(3, 4, 5)
        ax5.plot(x_pixels, before_amplitude, 'b-', label='Before Lens', linewidth=2)
        ax5.set_title('Amplitude: Before Lens')
        ax5.set_xlabel('Pixel')
        ax5.set_ylabel('Amplitude')
        ax5.grid(True, alpha=0.3)
        
        ax6 = plt.subplot(3, 4, 6)
        ax6.plot(x_pixels, after_amplitude, 'r-', label='After Lens', linewidth=2)
        ax6.set_title('Amplitude: After Lens')
        ax6.set_xlabel('Pixel')
        ax6.set_ylabel('Amplitude')
        ax6.grid(True, alpha=0.3)
        
        # Amplitude comparison
        ax7 = plt.subplot(3, 4, 7)
        ax7.plot(x_pixels, before_amplitude, 'b-', label='Before Lens', linewidth=2, alpha=0.7)
        ax7.plot(x_pixels, after_amplitude, 'r-', label='After Lens', linewidth=2, alpha=0.7)
        ax7.set_title('Amplitude Comparison')
        ax7.set_xlabel('Pixel')
        ax7.set_ylabel('Amplitude')
        ax7.legend()
        ax7.grid(True, alpha=0.3)
        
        # Sensor intensity profile
        ax8 = plt.subplot(3, 4, 8)
        sensor_profile = sensor_intensity[center_row, :]
        ax8.plot(x_pixels, sensor_profile, 'k-', label='Sensor Intensity', linewidth=2)
        ax8.set_title('Final Sensor Intensity')
        ax8.set_xlabel('Pixel')
        ax8.set_ylabel('Intensity')
        ax8.grid(True, alpha=0.3)
        
        # Row 3: Phase profiles
        ax9 = plt.subplot(3, 4, 9)
        ax9.plot(x_pixels, before_phase, 'b-', linewidth=2)
        ax9.set_title('Phase: Before Lens')
        ax9.set_xlabel('Pixel')
        ax9.set_ylabel('Phase (radians)')
        ax9.set_ylim(-np.pi, np.pi)
        ax9.grid(True, alpha=0.3)
        
        ax10 = plt.subplot(3, 4, 10)
        ax10.plot(x_pixels, after_phase, 'r-', linewidth=2)
        ax10.set_title('Phase: After Lens')
        ax10.set_xlabel('Pixel')
        ax10.set_ylabel('Phase (radians)')
        ax10.set_ylim(-np.pi, np.pi)
        ax10.grid(True, alpha=0.3)
        
        # Phase comparison
        ax11 = plt.subplot(3, 4, 11)
        ax11.plot(x_pixels, before_phase, 'b-', label='Before Lens', linewidth=2, alpha=0.7)
        ax11.plot(x_pixels, after_phase, 'r-', label='After Lens', linewidth=2, alpha=0.7)
        ax11.set_title('Phase Comparison')
        ax11.set_xlabel('Pixel')
        ax11.set_ylabel('Phase (radians)')
        ax11.set_ylim(-np.pi, np.pi)
        ax11.legend()
        ax11.grid(True, alpha=0.3)
        
        # Phase difference (lens effect)
        ax12 = plt.subplot(3, 4, 12)
        phase_diff = after_phase - before_phase
        # Wrap phase difference to [-π, π]
        phase_diff = np.angle(np.exp(1j * phase_diff))
        ax12.plot(x_pixels, phase_diff, 'g-', linewidth=2)
        ax12.set_title('Phase Added by Lens')
        ax12.set_xlabel('Pixel')
        ax12.set_ylabel('Phase Difference (radians)')
        ax12.set_ylim(-np.pi, np.pi)
        ax12.grid(True, alpha=0.3)
        
        plt.suptitle('550nm Imaging System: Complete Field Analysis', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
    else:
        # Fallback to simpler visualization if intermediate fields not available
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        
        axes[0].imshow(test_image.numpy(), cmap='gray')
        axes[0].set_title(f'Input: {image_type.value.title()}')
        axes[0].axis('off')
        
        sensor_intensity = result['intensity_sensor'].detach().numpy()
        axes[1].imshow(sensor_intensity, cmap='hot')
        axes[1].set_title('Sensor Output (Intensity)')
        axes[1].axis('off')
        
        center_row = sensor_intensity.shape[0] // 2
        input_profile = test_image[center_row, :].numpy()
        output_profile = sensor_intensity[center_row, :]
        
        axes[2].plot(input_profile, 'b-', label='Input', alpha=0.7, linewidth=2)
        axes[2].plot(output_profile, 'r-', label='Sensor Output', alpha=0.7, linewidth=2)
        axes[2].set_title('Center Cross-section')
        axes[2].set_xlabel('Pixel')
        axes[2].set_ylabel('Intensity')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        plt.suptitle('550nm Imaging System Simulation Results', fontsize=14, fontweight='bold')
        plt.tight_layout()
    
    # Save and/or display based on CLI arguments
    if script_manager.should_save():
        save_path = 'results/550nm_simulation_results.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        script_manager.record_result(save_path)
        script_manager.print_if_verbose(f"Simulation results saved to: {save_path}")
    
    if script_manager.should_display():
        script_manager.print_unless_quiet("Displaying simulation results...")
        plt.show()
    else:
        script_manager.print_unless_quiet("Simulation results saved to file")
        plt.close()
    
    script_manager.print_header("550NM IMAGING SYSTEM COMPLETE")
    script_manager.print_unless_quiet("System designed for 550nm wavelength")
    if script_manager.should_save():
        script_manager.print_unless_quiet("Auto-generated system layout saved to results/")
    script_manager.print_unless_quiet("Ready for imaging simulations")
    
    # Print summary
    script_manager.print_summary()
    
    return system, config

if __name__ == "__main__":
    system, config = build_550nm_imaging_system()