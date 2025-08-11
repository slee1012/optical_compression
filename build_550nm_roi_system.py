"""
Build a simple imaging system with 550nm design wavelength using ROI-based approach.

This version uses dynamic ROI tracking instead of upfront coordinate optimization.
The field size adapts automatically as information flows through the optical system.

Usage:
  python build_550nm_roi_system.py                    # Default behavior
  python build_550nm_roi_system.py display=True       # Show plots on screen
  python build_550nm_roi_system.py verbose=True       # Enable verbose output
  python build_550nm_roi_system.py save=False         # Disable saving
"""

import torch
from config.system_config import SystemConfig
from simulation.builder import SystemBuilder
from utils.cli_args import OpticalScriptManager
from utils.test_images import create_test_image, ImageType

def build_550nm_roi_imaging_system():
    # Setup CLI argument handling
    script_manager = OpticalScriptManager("Build 550nm ROI System")
    script_manager.print_header("BUILDING 550NM IMAGING SYSTEM (ROI-BASED)")
    
    # System parameters for 550nm green light
    design_wavelength = 550e-9  # 550nm green light
    focal_length = 25e-3         # 25mm focal length
    object_distance = 50e-3     # 50mm from object to lens
    aperture_diameter = 4e-3    # 4mm aperture
    
    # Create system configuration - ROI system will adapt field size dynamically
    config = SystemConfig(
        sensor_resolution=(512, 512),     # Starting resolution
        wavelength=design_wavelength,     # Design wavelength
        pixel_pitch=2e-6,                # Starting pixel pitch (2μm)
        propagation_distance=0.0         # Use explicit element positioning
    )
    
    script_manager.print_unless_quiet(f"Design wavelength: {design_wavelength*1e9:.0f}nm")
    script_manager.print_unless_quiet(f"Lens focal length: {focal_length*1000:.1f}mm")
    script_manager.print_unless_quiet(f"Object distance: {object_distance*1000:.1f}mm")
    script_manager.print_unless_quiet(f"Aperture diameter: {aperture_diameter*1000:.1f}mm")
    script_manager.print_unless_quiet(f"Initial sensor: {config.sensor_resolution[0]}x{config.sensor_resolution[1]} pixels")
    script_manager.print_unless_quiet(f"Initial pixel pitch: {config.pixel_pitch*1e6:.1f}μm")
    script_manager.print_unless_quiet(f"Initial field size: {config.sensor_resolution[0]*config.pixel_pitch*1e3:.2f}×{config.sensor_resolution[1]*config.pixel_pitch*1e3:.2f}mm")
    
    # Calculate theoretical image distance using lens equation
    # 1/f = 1/s_o + 1/s_i  ->  s_i = 1/(1/f - 1/s_o)
    s_i = 1 / (1/focal_length - 1/object_distance)
    theoretical_image_distance = object_distance + abs(s_i)
    
    script_manager.print_unless_quiet(f"\\nTheoretical image distance: {theoretical_image_distance*1000:.1f}mm")
    script_manager.print_unless_quiet(f"Magnification: {abs(s_i/object_distance):.3f}x")
    
    # Build the optical system with ROI-based approach
    script_manager.print_section("Building ROI-based optical system")
    
    # Setup visualization with CLI control
    from analysis.visualization import Visualizer
    visualizer = script_manager.setup_visualization()
    
    # Create ROI-based system (no coordinate optimization needed!)
    system = (SystemBuilder(config, auto_optimize_coordinates=False, use_roi_system=True)
              .add_lens(focal_length=focal_length,
                       position=object_distance,           # Lens at 50mm
                       spacing_after=abs(s_i),            # Distance to image
                       aperture_diameter=aperture_diameter)
              .build(auto_draw=script_manager.should_save()))  # Auto-draw based on save preference
    
    script_manager.print_unless_quiet(f"ROI-based system built with {len(system.element_specs)} optical element(s)")
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
    
    # Create test image at initial resolution (ROI system will adapt as needed)
    script_manager.print_unless_quiet(f"Creating {image_type.value} test image at {config.sensor_resolution[0]}×{config.sensor_resolution[1]}...")
    test_image = create_test_image(config.sensor_resolution, image_type)
    
    # Convert to complex field (amplitude with zero phase)
    test_field = torch.complex(test_image, torch.zeros_like(test_image))
    
    input_energy = (torch.abs(test_field)**2).sum()
    
    # Enable intermediate field caching to see ROI evolution
    system.set_cache_intermediates(True)
    
    # Propagate through the ROI-based system
    result = system(test_field)
    
    output_energy = result['intensity_sensor'].sum()
    energy_ratio = output_energy / input_energy
    
    script_manager.print_unless_quiet(f"Input energy: {input_energy:.3f}")
    script_manager.print_unless_quiet(f"Output energy: {output_energy:.3f}")
    script_manager.print_unless_quiet(f"Energy conservation: {energy_ratio:.3f}")
    script_manager.print_unless_quiet(f"Peak intensity: {result['intensity_sensor'].max():.3f}")
    script_manager.print_unless_quiet(f"SNR: {result['snr']}")
    
    # Display ROI evolution
    script_manager.print_section("ROI Evolution Analysis")
    if hasattr(system, 'print_roi_evolution'):
        system.print_roi_evolution()
    
    # Extract key intermediate fields for visualization
    before_lens_field = None
    after_lens_field = None
    
    if 'intermediate_labels' in result:
        for i, label in enumerate(result['intermediate_labels']):
            if 'Before ThinLens' in label:
                before_lens_field = {
                    'field': result['intermediate_fields'][i],
                    'position': 50.0,  # mm
                    'label': label
                }
            elif 'After ThinLens' in label:
                after_lens_field = {
                    'field': result['intermediate_fields'][i],
                    'position': 50.0,  # mm
                    'label': label
                }
    
    # Create and save simulation results visualization
    script_manager.print_section("Creating ROI-based simulation visualization")
    
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Create comprehensive visualization showing ROI evolution and field analysis
    if before_lens_field and after_lens_field:
        # Full analysis with before/after lens amplitude and phase + ROI info
        fig = plt.figure(figsize=(20, 16))
        
        # Row 1: ROI evolution visualization
        ax1 = plt.subplot(4, 4, 1)
        ax1.text(0.5, 0.5, f'Initial ROI\\n{config.sensor_resolution[0]}×{config.sensor_resolution[1]}\\n{config.pixel_pitch*1e6:.1f}μm pitch', 
                ha='center', va='center', transform=ax1.transAxes, fontsize=10)
        ax1.set_title('Initial ROI')
        ax1.axis('off')
        
        if 'final_roi' in result:
            final_roi = result['final_roi']
            ax2 = plt.subplot(4, 4, 2)
            ax2.text(0.5, 0.5, f'Final ROI\\n{final_roi.resolution[0]}×{final_roi.resolution[1]}\\n{final_roi.pixel_pitch*1e6:.1f}μm pitch', 
                    ha='center', va='center', transform=ax2.transAxes, fontsize=10)
            ax2.set_title('Final ROI')
            ax2.axis('off')
            
            # ROI expansion factor
            expansion_factor = final_roi.resolution[0] / config.sensor_resolution[0]
            ax3 = plt.subplot(4, 4, 3)
            ax3.text(0.5, 0.5, f'ROI Expansion\\n{expansion_factor:.1f}×', 
                    ha='center', va='center', transform=ax3.transAxes, fontsize=12, fontweight='bold')
            ax3.set_title('Dynamic Adaptation')
            ax3.axis('off')
        
        # Energy conservation
        ax4 = plt.subplot(4, 4, 4)
        ax4.text(0.5, 0.5, f'Energy Conservation\\n{energy_ratio:.3f}', 
                ha='center', va='center', transform=ax4.transAxes, fontsize=12, fontweight='bold', 
                color='green' if energy_ratio > 0.99 else 'orange')
        ax4.set_title('Conservation Check')
        ax4.axis('off')
        
        # Row 2: 2D intensity images
        ax5 = plt.subplot(4, 4, 5)
        ax5.imshow(test_image.numpy(), cmap='gray')
        ax5.set_title(f'Input: {image_type.value.title()}')
        ax5.axis('off')
        
        ax6 = plt.subplot(4, 4, 6)
        before_lens_intensity = (torch.abs(before_lens_field['field'])**2).detach().numpy()
        ax6.imshow(before_lens_intensity, cmap='viridis')
        ax6.set_title('Before Lens (50.0mm)')
        ax6.axis('off')
        
        ax7 = plt.subplot(4, 4, 7)
        after_lens_intensity = (torch.abs(after_lens_field['field'])**2).detach().numpy()
        ax7.imshow(after_lens_intensity, cmap='viridis')
        ax7.set_title('After Lens (50.0mm)')
        ax7.axis('off')
        
        ax8 = plt.subplot(4, 4, 8)
        sensor_intensity = result['intensity_sensor'].detach().numpy()
        ax8.imshow(sensor_intensity, cmap='hot')
        ax8.set_title('Final Sensor Output')
        ax8.axis('off')
        
        # Row 3: Amplitude profiles
        center_row_input = test_image.shape[0] // 2
        center_row_before = before_lens_intensity.shape[0] // 2
        center_row_after = after_lens_intensity.shape[0] // 2
        center_row_sensor = sensor_intensity.shape[0] // 2
        
        x_pixels_input = np.arange(test_image.shape[1])
        x_pixels_before = np.arange(before_lens_intensity.shape[1])
        x_pixels_after = np.arange(after_lens_intensity.shape[1])
        x_pixels_sensor = np.arange(sensor_intensity.shape[1])
        
        # Extract complex field data for amplitude analysis
        before_lens_complex = before_lens_field['field'][center_row_before, :].detach()
        after_lens_complex = after_lens_field['field'][center_row_after, :].detach()
        
        before_amplitude = torch.abs(before_lens_complex).numpy()
        after_amplitude = torch.abs(after_lens_complex).numpy()
        
        ax9 = plt.subplot(4, 4, 9)
        ax9.plot(x_pixels_input, test_image[center_row_input, :].numpy(), 'k-', linewidth=2, label='Input')
        ax9.set_title('Input Amplitude Profile')
        ax9.set_xlabel('Pixel')
        ax9.set_ylabel('Amplitude')
        ax9.grid(True, alpha=0.3)
        
        ax10 = plt.subplot(4, 4, 10)
        ax10.plot(x_pixels_before, before_amplitude, 'b-', linewidth=2)
        ax10.set_title('Before Lens Amplitude')
        ax10.set_xlabel('Pixel')
        ax10.set_ylabel('Amplitude')
        ax10.grid(True, alpha=0.3)
        
        ax11 = plt.subplot(4, 4, 11)
        ax11.plot(x_pixels_after, after_amplitude, 'r-', linewidth=2)
        ax11.set_title('After Lens Amplitude')
        ax11.set_xlabel('Pixel')
        ax11.set_ylabel('Amplitude')
        ax11.grid(True, alpha=0.3)
        
        ax12 = plt.subplot(4, 4, 12)
        ax12.plot(x_pixels_sensor, sensor_intensity[center_row_sensor, :], 'g-', linewidth=2)
        ax12.set_title('Sensor Intensity Profile')
        ax12.set_xlabel('Pixel')
        ax12.set_ylabel('Intensity')
        ax12.grid(True, alpha=0.3)
        
        # Row 4: Phase analysis
        before_phase = torch.angle(before_lens_complex).numpy()
        after_phase = torch.angle(after_lens_complex).numpy()
        
        ax13 = plt.subplot(4, 4, 13)
        ax13.plot(x_pixels_before, before_phase, 'b-', linewidth=2)
        ax13.set_title('Phase: Before Lens')
        ax13.set_xlabel('Pixel')
        ax13.set_ylabel('Phase (radians)')
        ax13.set_ylim(-np.pi, np.pi)
        ax13.grid(True, alpha=0.3)
        
        ax14 = plt.subplot(4, 4, 14)
        ax14.plot(x_pixels_after, after_phase, 'r-', linewidth=2)
        ax14.set_title('Phase: After Lens')
        ax14.set_xlabel('Pixel')
        ax14.set_ylabel('Phase (radians)')
        ax14.set_ylim(-np.pi, np.pi)
        ax14.grid(True, alpha=0.3)
        
        # Phase difference (lens effect)
        if before_phase.shape == after_phase.shape:
            phase_diff = after_phase - before_phase
            phase_diff = np.angle(np.exp(1j * phase_diff))  # Wrap to [-π, π]
            ax15 = plt.subplot(4, 4, 15)
            ax15.plot(x_pixels_after, phase_diff, 'g-', linewidth=2)
            ax15.set_title('Phase Added by Lens')
            ax15.set_xlabel('Pixel')
            ax15.set_ylabel('Phase Difference (radians)')
            ax15.set_ylim(-np.pi, np.pi)
            ax15.grid(True, alpha=0.3)
        
        # ROI evolution timeline
        ax16 = plt.subplot(4, 4, 16)
        if 'roi_history' in result:
            roi_sizes = [roi.resolution[0] for roi in result['roi_history']]
            roi_steps = list(range(len(roi_sizes)))
            ax16.plot(roi_steps, roi_sizes, 'o-', linewidth=2, markersize=6)
            ax16.set_title('ROI Size Evolution')
            ax16.set_xlabel('Propagation Step')
            ax16.set_ylabel('Resolution')
            ax16.grid(True, alpha=0.3)
        
        plt.suptitle('550nm ROI-Based Imaging System: Complete Analysis', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
    else:
        # Fallback to simpler visualization
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        
        axes[0].imshow(test_image.numpy(), cmap='gray')
        axes[0].set_title(f'Input: {image_type.value.title()}')
        axes[0].axis('off')
        
        sensor_intensity = result['intensity_sensor'].detach().numpy()
        axes[1].imshow(sensor_intensity, cmap='hot')
        axes[1].set_title('ROI-Based Sensor Output')
        axes[1].axis('off')
        
        center_row = sensor_intensity.shape[0] // 2
        input_profile = test_image[test_image.shape[0]//2, :].numpy()
        output_profile = sensor_intensity[center_row, :]
        
        axes[2].plot(input_profile, 'b-', label='Input', alpha=0.7, linewidth=2)
        axes[2].plot(output_profile, 'r-', label='ROI Output', alpha=0.7, linewidth=2)
        axes[2].set_title('Cross-section Comparison')
        axes[2].set_xlabel('Pixel')
        axes[2].set_ylabel('Intensity')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        plt.suptitle('550nm ROI-Based Imaging System Results', fontsize=14, fontweight='bold')
        plt.tight_layout()
    
    # Save and/or display based on CLI arguments
    if script_manager.should_save():
        save_path = 'results/550nm_roi_simulation_results.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        script_manager.record_result(save_path)
        script_manager.print_if_verbose(f"ROI simulation results saved to: {save_path}")
    
    if script_manager.should_display():
        script_manager.print_unless_quiet("Displaying ROI-based simulation results...")
        plt.show()
    else:
        script_manager.print_unless_quiet("ROI simulation results saved to file")
        plt.close()
    
    script_manager.print_header("550NM ROI-BASED IMAGING SYSTEM COMPLETE")
    script_manager.print_unless_quiet("System designed for 550nm wavelength with dynamic ROI tracking")
    script_manager.print_unless_quiet("Field size automatically adapted during propagation")
    if script_manager.should_save():
        script_manager.print_unless_quiet("Auto-generated system layout saved to results/")
    script_manager.print_unless_quiet("Ready for advanced optical simulations")
    
    # Print summary
    script_manager.print_summary()
    
    return system, config

if __name__ == "__main__":
    system, config = build_550nm_roi_imaging_system()