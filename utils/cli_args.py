"""
Command-line argument utilities for optical simulation scripts.

Provides standardized CLI arguments for:
- Display control (show plots)
- Output control (save results)
- Visualization options
- System configuration overrides
"""

import argparse
import sys
from typing import Dict, Any, Optional

def add_display_args(parser: argparse.ArgumentParser):
    """Add display-related arguments to parser."""
    display_group = parser.add_argument_group('Display Options')
    display_group.add_argument(
        '--display', '--show', 
        action='store_true', 
        default=False,
        help='Display plots and visualizations on screen'
    )
    display_group.add_argument(
        '--no-save', 
        action='store_true', 
        default=False,
        help='Disable automatic saving of results'
    )
    display_group.add_argument(
        '--save-dir', 
        type=str, 
        default='results',
        help='Directory to save results (default: results)'
    )
    return parser

def add_system_args(parser: argparse.ArgumentParser):
    """Add system configuration override arguments."""
    system_group = parser.add_argument_group('System Configuration')
    system_group.add_argument(
        '--resolution', 
        type=int, 
        nargs=2, 
        metavar=('H', 'W'),
        help='Override sensor resolution (height width)'
    )
    system_group.add_argument(
        '--pixel-pitch', 
        type=float, 
        metavar='PITCH',
        help='Override pixel pitch in micrometers'
    )
    system_group.add_argument(
        '--wavelength', 
        type=float, 
        metavar='WL',
        help='Override wavelength in nanometers'
    )
    system_group.add_argument(
        '--optimize-coords', 
        action='store_true', 
        default=None,
        help='Force coordinate optimization'
    )
    system_group.add_argument(
        '--no-optimize-coords', 
        action='store_true', 
        default=False,
        help='Disable coordinate optimization'
    )
    return parser

def add_verbose_args(parser: argparse.ArgumentParser):
    """Add verbosity control arguments."""
    verbose_group = parser.add_argument_group('Verbosity Control')
    verbose_group.add_argument(
        '--verbose', '-v', 
        action='store_true', 
        default=False,
        help='Enable verbose output'
    )
    verbose_group.add_argument(
        '--quiet', '-q', 
        action='store_true', 
        default=False,
        help='Suppress non-essential output'
    )
    return parser

def create_standard_parser(description: str = "Optical simulation script") -> argparse.ArgumentParser:
    """Create parser with standard optical simulation arguments."""
    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Add all standard argument groups
    add_display_args(parser)
    add_system_args(parser)
    add_verbose_args(parser)
    
    return parser

def parse_display_args() -> Dict[str, Any]:
    """
    Quick parser for display arguments from sys.argv.
    
    Supports simple key=value format:
    python script.py display=True save=False verbose=True
    
    Returns:
        Dict with parsed arguments
    """
    args = {
        'display': False,
        'save': True,
        'verbose': False,
        'quiet': False,
        'optimize_coords': None
    }
    
    for arg in sys.argv[1:]:
        if '=' in arg:
            key, value = arg.split('=', 1)
            key = key.lower().replace('-', '_')
            
            # Convert string values to appropriate types
            if value.lower() in ('true', '1', 'yes', 'on'):
                value = True
            elif value.lower() in ('false', '0', 'no', 'off'):
                value = False
            elif value.isdigit():
                value = int(value)
            elif key in ('pixel_pitch', 'wavelength'):
                try:
                    value = float(value)
                except ValueError:
                    pass
            
            args[key] = value
    
    return args

def update_config_from_args(config, args: Dict[str, Any]):
    """Update SystemConfig based on parsed arguments."""
    if args.get('resolution') and len(args['resolution']) == 2:
        config.sensor_resolution = tuple(args['resolution'])
    
    if args.get('pixel_pitch') is not None:
        # Convert from micrometers to meters
        config.pixel_pitch = args['pixel_pitch'] * 1e-6
    
    if args.get('wavelength') is not None:
        # Convert from nanometers to meters
        config.wavelength = args['wavelength'] * 1e-9
    
    return config

class OpticalScriptManager:
    """
    Helper class to manage optical simulation scripts with CLI arguments.
    
    Handles display control, saving, and provides consistent interface.
    """
    
    def __init__(self, script_name: str = ""):
        self.script_name = script_name
        self.args = parse_display_args()
        self.results_saved = []
    
    def should_display(self) -> bool:
        """Check if plots should be displayed."""
        return self.args.get('display', False)
    
    def should_save(self) -> bool:
        """Check if results should be saved."""
        return self.args.get('save', True)
    
    def is_verbose(self) -> bool:
        """Check if verbose output is enabled."""
        return self.args.get('verbose', False)
    
    def is_quiet(self) -> bool:
        """Check if quiet mode is enabled."""
        return self.args.get('quiet', False)
    
    def should_optimize_coords(self) -> Optional[bool]:
        """Check coordinate optimization preference."""
        if self.args.get('optimize_coords') is True:
            return True
        elif self.args.get('no_optimize_coords') is True:
            return False
        return None  # Use default
    
    def print_if_verbose(self, *args, **kwargs):
        """Print only if verbose mode is enabled."""
        if self.is_verbose():
            print(*args, **kwargs)
    
    def print_unless_quiet(self, *args, **kwargs):
        """Print unless quiet mode is enabled."""
        if not self.is_quiet():
            print(*args, **kwargs)
    
    def setup_visualization(self, visualizer=None):
        """Setup visualization based on CLI arguments."""
        if visualizer is None:
            from analysis.visualization import Visualizer
            visualizer = Visualizer()
        
        # Monkey patch visualization methods to respect display settings
        original_show = getattr(visualizer, '_original_show', None)
        if original_show is None:
            # Store original methods
            for method_name in ['draw_optical_system_geometry', 'draw_imaging_system_layout', 
                              'plot_compression_pipeline', 'plot_metrics', 'plot_optical_elements']:
                if hasattr(visualizer, method_name):
                    method = getattr(visualizer, method_name)
                    setattr(visualizer, f'_original_{method_name}', method)
        
        # Override show parameter based on CLI args
        def wrap_viz_method(method_name):
            original_method = getattr(visualizer, f'_original_{method_name}', None)
            if original_method is None:
                return
                
            def wrapper(*args, **kwargs):
                # Override show parameter
                kwargs['show'] = self.should_display()
                if not self.should_save():
                    kwargs['save_path'] = None
                return original_method(*args, **kwargs)
            
            setattr(visualizer, method_name, wrapper)
        
        # Wrap visualization methods
        for method_name in ['draw_optical_system_geometry', 'draw_imaging_system_layout',
                           'plot_compression_pipeline', 'plot_metrics', 'plot_optical_elements']:
            wrap_viz_method(method_name)
        
        return visualizer
    
    def print_header(self, title: str):
        """Print formatted header."""
        if not self.is_quiet():
            print("=" * 80)
            print(title)
            print("=" * 80)
    
    def print_section(self, title: str):
        """Print formatted section header.""" 
        if not self.is_quiet():
            print(f"\\n{title}")
            print("-" * len(title))
    
    def record_result(self, filepath: str):
        """Record a saved result file."""
        self.results_saved.append(filepath)
        if self.is_verbose():
            print(f"Saved: {filepath}")
    
    def print_summary(self):
        """Print summary of script execution."""
        if not self.is_quiet():
            print("\\n" + "=" * 80)
            print("SCRIPT EXECUTION SUMMARY")
            if self.results_saved:
                print(f"Results saved ({len(self.results_saved)} files):")
                for filepath in self.results_saved:
                    print(f"  - {filepath}")
            else:
                print("No results saved (saving disabled)")
            
            if self.should_display():
                print("Visualizations displayed on screen")
            else:
                print("Visualizations saved to files only")
            print("=" * 80)