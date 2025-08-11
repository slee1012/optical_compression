"""
Demo script showing CLI argument features for optical simulations.

Usage Examples:
  python demo_cli_features.py                     # Default behavior
  python demo_cli_features.py display=True        # Show plots on screen  
  python demo_cli_features.py verbose=True        # Enable verbose output
  python demo_cli_features.py quiet=True          # Minimal output
  python demo_cli_features.py save=False          # Disable saving
  python demo_cli_features.py display=True save=False verbose=True  # Combine options
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
from config.system_config import SystemConfig
from utils.cli_args import OpticalScriptManager

def demo_cli_features():
    """Demonstrate CLI argument features."""
    # Setup CLI management
    script_manager = OpticalScriptManager("CLI Demo")
    
    # Headers and sections respond to quiet mode
    script_manager.print_header("CLI FEATURES DEMONSTRATION")
    script_manager.print_section("Testing CLI Argument Parsing")
    
    # Different output types
    script_manager.print_unless_quiet("This message respects quiet mode")
    script_manager.print_if_verbose("This message only shows if verbose=True")
    
    # Show parsed arguments
    if script_manager.is_verbose():
        print(f"Parsed CLI arguments: {script_manager.args}")
    
    # Create some dummy data for visualization demo
    script_manager.print_section("Creating Test Visualization")
    
    # Create test data
    x, y = np.meshgrid(np.linspace(-2, 2, 64), np.linspace(-2, 2, 64))
    test_data = np.exp(-(x**2 + y**2)) * np.sin(3*x) * np.cos(3*y)
    
    # Create a plot (respects display settings)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    
    # Plot 1: Original data
    ax1.imshow(test_data, cmap='viridis')
    ax1.set_title('Test Pattern')
    ax1.axis('off')
    
    # Plot 2: FFT
    fft_data = np.abs(np.fft.fftshift(np.fft.fft2(test_data)))
    ax2.imshow(fft_data, cmap='plasma')
    ax2.set_title('FFT Magnitude')
    ax2.axis('off')
    
    plt.tight_layout()
    
    # Save/display based on CLI arguments
    if script_manager.should_save():
        save_path = 'results/cli_demo_visualization.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        script_manager.record_result(save_path)
        script_manager.print_if_verbose(f"Visualization saved to: {save_path}")
    
    if script_manager.should_display():
        script_manager.print_unless_quiet("Displaying plot on screen...")
        plt.show()
    else:
        script_manager.print_unless_quiet("Plot saved to file (not displayed)")
        plt.close()
    
    # Demo system configuration override
    script_manager.print_section("System Configuration Features")
    
    config = SystemConfig(
        sensor_resolution=(128, 128),
        wavelength=550e-9,
        pixel_pitch=5e-6
    )
    
    # Show how CLI args could override config (if implemented)
    script_manager.print_unless_quiet(f"Base config - Resolution: {config.sensor_resolution}")
    script_manager.print_unless_quiet(f"Base config - Wavelength: {config.wavelength*1e9:.0f}nm")
    script_manager.print_unless_quiet(f"Base config - Pixel pitch: {config.pixel_pitch*1e6:.1f}μm")
    
    if 'wavelength' in script_manager.args:
        script_manager.print_unless_quiet(f"CLI override - Wavelength: {script_manager.args['wavelength']}nm")
    
    # Demo coordinate optimization control
    script_manager.print_section("Coordinate Optimization Control")
    optimize_pref = script_manager.should_optimize_coords()
    if optimize_pref is True:
        script_manager.print_unless_quiet("Coordinate optimization: ENABLED (via CLI)")
    elif optimize_pref is False:
        script_manager.print_unless_quiet("Coordinate optimization: DISABLED (via CLI)")
    else:
        script_manager.print_unless_quiet("Coordinate optimization: DEFAULT")
    
    # Summary
    script_manager.print_header("CLI FEATURES SUMMARY")
    script_manager.print_unless_quiet("✓ Display control: display=True/False".replace('✓', '+'))
    script_manager.print_unless_quiet("✓ Save control: save=True/False".replace('✓', '+'))
    script_manager.print_unless_quiet("✓ Verbosity: verbose=True, quiet=True".replace('✓', '+'))
    script_manager.print_unless_quiet("✓ System config overrides: wavelength=X, pixel_pitch=X".replace('✓', '+'))
    script_manager.print_unless_quiet("✓ Coordinate optimization: optimize_coords=True/False".replace('✓', '+'))
    
    # Print execution summary
    script_manager.print_summary()

if __name__ == "__main__":
    import os
    os.makedirs('results', exist_ok=True)
    demo_cli_features()