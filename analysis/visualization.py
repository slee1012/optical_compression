import torch
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict, Optional


class Visualizer:
    
    def __init__(self, figsize=(15, 10)):
        self.figsize = figsize
    
    def plot_compression_pipeline(self, original, sensor_output, reconstructed=None, 
                                 intermediate_fields=None, save_path='auto'):
        
        # Prepare data
        def prepare_image(tensor):
            if torch.is_complex(tensor):
                return torch.abs(tensor).cpu().numpy(), torch.angle(tensor).cpu().numpy()
            else:
                return tensor.cpu().numpy(), None
        
        # Determine number of plots
        n_plots = 2 + (1 if reconstructed is not None else 0)
        if intermediate_fields:
            n_plots += min(len(intermediate_fields), 2)
        
        fig, axes = plt.subplots(1, n_plots, figsize=(n_plots * 4, 4))
        
        if n_plots == 1:
            axes = [axes]
        
        idx = 0
        
        # Original image
        img, _ = prepare_image(original)
        axes[idx].imshow(img, cmap='gray')
        axes[idx].set_title('Original Image')
        axes[idx].axis('off')
        idx += 1
        
        # Sensor output
        img, _ = prepare_image(sensor_output)
        axes[idx].imshow(img, cmap='gray')
        axes[idx].set_title('Sensor Output')
        axes[idx].axis('off')
        idx += 1
        
        # Reconstructed (if available)
        if reconstructed is not None:
            img, _ = prepare_image(reconstructed)
            axes[idx].imshow(img, cmap='gray')
            axes[idx].set_title('Reconstructed')
            axes[idx].axis('off')
            idx += 1
        
        # Intermediate fields (show first 2)
        if intermediate_fields:
            for i, field in enumerate(intermediate_fields[:2]):
                if idx >= len(axes):
                    break
                mag, phase = prepare_image(field[0] if field.dim() > 2 else field)
                
                if phase is not None:
                    axes[idx].imshow(phase, cmap='hsv', vmin=-np.pi, vmax=np.pi)
                    axes[idx].set_title(f'Field {i+1} Phase')
                else:
                    axes[idx].imshow(mag, cmap='gray')
                    axes[idx].set_title(f'Field {i+1}')
                
                axes[idx].axis('off')
                idx += 1
        
        plt.suptitle('Optical Compression Pipeline', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        # Always save visualization
        import os
        os.makedirs('results', exist_ok=True)
        
        if save_path == 'auto':
            # Generate automatic filename
            n_plots = 2 + (1 if reconstructed is not None else 0)
            if intermediate_fields:
                n_plots += min(len(intermediate_fields), 2)
            save_path = f'results/compression_pipeline_{n_plots}plots.png'
        elif save_path and save_path != 'auto':
            # Use provided path
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Pipeline visualization saved to: {save_path}")
        plt.show()
    
    def plot_metrics(self, metrics_history, save_path='auto'):
        fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        
        metrics_names = ['mse', 'psnr', 'compression_ratio', 'frequency_error']
        
        for ax, metric in zip(axes.flat, metrics_names):
            if metric in metrics_history[0]:
                values = [m[metric] for m in metrics_history]
                ax.plot(values)
                ax.set_title(metric.upper())
                ax.set_xlabel('Iteration')
                ax.grid(True)
        
        plt.tight_layout()
        
        # Always save visualization
        import os
        os.makedirs('results', exist_ok=True)
        
        if save_path == 'auto':
            # Generate automatic filename
            n_iterations = len(metrics_history)
            save_path = f'results/metrics_history_{n_iterations}iters.png'
        elif save_path and save_path != 'auto':
            # Use provided path
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Metrics plot saved to: {save_path}")
        plt.show()
    
    def plot_optical_elements(self, elements, save_path='auto'):
        n_elements = len(elements)
        fig, axes = plt.subplots(1, n_elements, figsize=(n_elements * 4, 4))
        
        if n_elements == 1:
            axes = [axes]
        
        for i, element in enumerate(elements):
            viz_data = element.visualize() if hasattr(element, 'visualize') else {}
            
            if 'phase' in viz_data:
                phase = viz_data['phase'].cpu().numpy()
                im = axes[i].imshow(phase, cmap='hsv', vmin=-np.pi, vmax=np.pi)
                axes[i].set_title(f'{element.name}\nPhase')
            elif 'amplitude' in viz_data:
                amp = viz_data['amplitude'].cpu().numpy()
                im = axes[i].imshow(amp, cmap='gray')
                axes[i].set_title(f'{element.name}\nAmplitude')
            
            axes[i].axis('off')
        
        plt.suptitle('Optical Elements', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        # Always save visualization
        import os
        os.makedirs('results', exist_ok=True)
        
        if save_path == 'auto':
            # Generate automatic filename
            n_elements = len(elements)
            element_types = '_'.join([elem.__class__.__name__ for elem in elements[:3]])
            if len(elements) > 3:
                element_types += f'_plus{len(elements)-3}more'
            save_path = f'results/optical_elements_{element_types}_{n_elements}elem.png'
        elif save_path and save_path != 'auto':
            # Use provided path
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Elements visualization saved to: {save_path}")
        plt.show()
    
    def draw_imaging_system_layout(self, object_distance, lens_position, sensor_positions, 
                                 focal_length, show=True, save_path='auto'):
        """
        Draw the imaging system layout showing object, lens, and sensor positions.
        
        Args:
            object_distance: Distance from object to lens (meters)
            lens_position: Position of lens relative to object (meters)
            sensor_positions: List of sensor positions (meters from object)
            focal_length: Lens focal length (meters)
            show: Whether to display the plot
            save_path: Path to save the figure ('auto' for automatic naming)
            
        Returns:
            matplotlib Figure object
        """
        fig, ax = plt.subplots(1, 1, figsize=(14, 6))
        
        # Convert to mm for display
        obj_pos = 0
        lens_pos = lens_position * 1000  # mm
        sensor_pos_mm = [pos * 1000 for pos in sensor_positions]  # mm
        f_mm = focal_length * 1000  # mm
        
        # Calculate theoretical focus position
        # 1/f = 1/s_o + 1/s_i, where s_o = object distance, s_i = image distance
        s_o = lens_position  # object distance from lens
        s_i_theoretical = 1 / (1/focal_length - 1/s_o) if (1/focal_length - 1/s_o) != 0 else 1e6
        theoretical_focus = (lens_position + s_i_theoretical) * 1000  # mm from object
        
        # Draw optical axis
        max_pos = max(sensor_pos_mm) + 5
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5, label='Optical Axis')
        
        # Draw object
        ax.axvline(x=obj_pos, color='blue', linewidth=4, alpha=0.8, label='Object')
        ax.text(obj_pos, 15, 'Object', ha='center', va='bottom', fontsize=10, color='blue')
        
        # Draw lens
        lens_height = 12
        ax.plot([lens_pos, lens_pos], [-lens_height, lens_height], 
               'red', linewidth=6, alpha=0.8, label=f'Lens (f={f_mm:.0f}mm)')
        ax.text(lens_pos, lens_height + 2, f'Lens\\nf={f_mm:.0f}mm', 
               ha='center', va='bottom', fontsize=10, color='red')
        
        # Draw lens symbol (curved lines)
        curve_x = np.linspace(lens_pos-0.5, lens_pos+0.5, 20)
        curve_y_top = 12 - 0.3 * (curve_x - lens_pos)**2 / 0.25
        curve_y_bottom = -12 + 0.3 * (curve_x - lens_pos)**2 / 0.25
        ax.plot(curve_x, curve_y_top, 'red', linewidth=2, alpha=0.6)
        ax.plot(curve_x, curve_y_bottom, 'red', linewidth=2, alpha=0.6)
        
        # Draw sensor positions
        colors = ['green', 'orange', 'purple', 'brown', 'pink']
        for i, (pos, color) in enumerate(zip(sensor_pos_mm, colors)):
            ax.axvline(x=pos, color=color, linewidth=4, alpha=0.8, 
                      label=f'Sensor {i+1} ({pos:.0f}mm)')
            ax.text(pos, -15, f'Sensor {i+1}\\n{pos:.0f}mm', 
                   ha='center', va='top', fontsize=9, color=color)
        
        # Draw theoretical focus position
        if abs(theoretical_focus) < 1000:  # Only show if reasonable
            ax.axvline(x=theoretical_focus, color='red', linestyle=':', linewidth=2, alpha=0.7,
                      label=f'Theoretical Focus ({theoretical_focus:.1f}mm)')
            ax.text(theoretical_focus, -20, f'Theory\\nFocus\\n{theoretical_focus:.1f}mm', 
                   ha='center', va='top', fontsize=8, color='red')
        
        # Draw light rays
        ray_heights = [-8, -4, 0, 4, 8]
        for h in ray_heights:
            # From object to lens
            ax.plot([obj_pos, lens_pos], [h, h*0.8], 'b-', alpha=0.3, linewidth=1)
            
            # From lens to sensors (show convergence/divergence)
            for pos in sensor_pos_mm:
                # Calculate ray angle after lens (simplified)
                angle_factor = h * 0.8 / lens_pos if lens_pos != 0 else 0
                ray_end = h*0.8 - angle_factor * (pos - lens_pos) * 0.5
                ax.plot([lens_pos, pos], [h*0.8, ray_end], 'g-', alpha=0.2, linewidth=1)
        
        # Add dimensions
        ax.annotate('', xy=(obj_pos, -25), xytext=(lens_pos, -25),
                   arrowprops=dict(arrowstyle='<->', color='gray'))
        ax.text(lens_pos/2, -27, f'{lens_pos:.0f}mm', ha='center', va='top', fontsize=9, color='gray')
        
        # Formatting
        ax.set_xlim(-2, max_pos + 2)
        ax.set_ylim(-30, 20)
        ax.set_xlabel('Distance from Object (mm)', fontsize=12)
        ax.set_ylabel('Height (mm)', fontsize=12)
        ax.set_title('Imaging System Layout', fontsize=14, weight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Add lens equation info
        info_text = f"""Lens Equation:
1/f = 1/s₀ + 1/sᵢ
f = {f_mm:.0f}mm
s₀ = {lens_pos:.0f}mm (object distance)
sᵢ = {s_i_theoretical*1000:.1f}mm (image distance)
Focus at {theoretical_focus:.1f}mm from object"""
        
        ax.text(0.02, 0.98, info_text, transform=ax.transAxes, fontsize=9,
               verticalalignment='top', 
               bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        plt.tight_layout()
        
        # Always save visualization
        import os
        os.makedirs('results', exist_ok=True)
        
        if save_path == 'auto':
            # Generate automatic filename based on system parameters
            f_mm = focal_length * 1000
            save_path = f'results/imaging_system_layout_f{f_mm:.0f}mm.png'
        elif save_path and save_path != 'auto':
            # Use provided path
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Layout saved to: {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
        
        return fig
    
    def draw_optical_system_geometry(self, elements, element_positions, element_spacings=None,
                                   element_names=None, show=False, save_path='auto', include_imaging=True,
                                   system_config=None):
        """
        Draw a 2D geometric layout of an optical system showing all elements.
        
        Args:
            elements: List of optical elements
            element_positions: List of element positions (meters)
            element_spacings: Optional list of spacings between elements 
            element_names: Optional list of element names
            show: Whether to display the plot (default False to avoid blocking tests)
            save_path: Optional path to save the figure
            include_imaging: Whether to automatically add object and sensor positions for lens systems
            system_config: Optional SystemConfig for sensor size and image dimensions
            
        Returns:
            matplotlib Figure object
        """
        fig, ax = plt.subplots(1, 1, figsize=(16, 8))
        
        # Convert positions to mm
        positions_mm = [pos * 1000 for pos in element_positions]
        
        # Colors for different element types
        element_colors = {
            'ThinLens': 'red',
            'ThickLens': 'darkred', 
            'PhaseMask': 'blue',
            'CodedAperture': 'green',
            'LensArray': 'orange',
            'Waveguide': 'purple'
        }
        
        # Draw optical axis
        max_pos = max(positions_mm) + 10
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5, label='Optical Axis')
        
        # Draw elements
        for i, (element, pos_mm) in enumerate(zip(elements, positions_mm)):
            element_name = element_names[i] if element_names else element.__class__.__name__
            element_type = element.__class__.__name__
            color = element_colors.get(element_type, 'black')
            
            # Draw element representation
            if 'Lens' in element_type:
                # Draw lens
                lens_height = 15
                ax.plot([pos_mm, pos_mm], [-lens_height, lens_height], 
                       color=color, linewidth=6, alpha=0.8, label=f'{element_name}')
                
                # Add lens curves
                curve_x = np.linspace(pos_mm-1, pos_mm+1, 20)
                curve_y_top = lens_height - 0.5 * (curve_x - pos_mm)**2
                curve_y_bottom = -lens_height + 0.5 * (curve_x - pos_mm)**2
                ax.plot(curve_x, curve_y_top, color=color, linewidth=2, alpha=0.6)
                ax.plot(curve_x, curve_y_bottom, color=color, linewidth=2, alpha=0.6)
                
                # Add focal length info if available
                if hasattr(element, 'focal_length'):
                    f_mm = element.focal_length * 1000
                    ax.text(pos_mm, lens_height + 3, f'{element_name}\\nf={f_mm:.1f}mm',
                           ha='center', va='bottom', fontsize=10, color=color)
                else:
                    ax.text(pos_mm, lens_height + 3, element_name,
                           ha='center', va='bottom', fontsize=10, color=color)
                    
            else:
                # Draw other elements as vertical lines
                element_height = 10
                ax.plot([pos_mm, pos_mm], [-element_height, element_height], 
                       color=color, linewidth=4, alpha=0.8, label=f'{element_name}')
                ax.text(pos_mm, element_height + 2, element_name,
                       ha='center', va='bottom', fontsize=10, color=color)
        
        # Add object and sensor positions for lens systems
        if include_imaging:
            lens_elements = [(i, elem) for i, elem in enumerate(elements) if hasattr(elem, 'focal_length')]
            if lens_elements:
                lens_idx, lens_element = lens_elements[0]  # Use first lens
                lens_pos_mm = positions_mm[lens_idx]
                focal_length = lens_element.focal_length
                
                # Calculate object and image sizes if config available
                object_size_mm = None
                sensor_size_mm = None
                image_size_mm = None
                
                if system_config:
                    # Sensor physical dimensions
                    sensor_height = system_config.sensor_resolution[0] * system_config.pixel_pitch
                    sensor_width = system_config.sensor_resolution[1] * system_config.pixel_pitch
                    sensor_size_mm = max(sensor_height, sensor_width) * 1000  # mm
                    
                    # Calculate object and image sizes using magnification
                    object_distance = element_positions[lens_idx]
                    try:
                        s_i = 1 / (1/focal_length - 1/object_distance)
                        magnification = abs(s_i / object_distance)
                        object_size_mm = sensor_size_mm / magnification if magnification > 0 else sensor_size_mm
                        image_size_mm = sensor_size_mm  # Image size matches sensor
                    except:
                        object_size_mm = sensor_size_mm
                        image_size_mm = sensor_size_mm
                
                # Object at z=0
                object_pos_mm = 0
                if object_size_mm:
                    # Draw object with size
                    obj_half_height = object_size_mm / 4  # Make it visible
                    ax.plot([object_pos_mm, object_pos_mm], [-obj_half_height, obj_half_height], 
                           color='blue', linewidth=6, alpha=0.8, label='Object')
                    ax.text(object_pos_mm, obj_half_height + 2, f'Object\\n{object_size_mm:.1f}mm', 
                           ha='center', va='bottom', fontsize=10, color='blue')
                else:
                    ax.axvline(x=object_pos_mm, color='blue', linewidth=4, alpha=0.8, label='Object')
                    ax.text(object_pos_mm, 18, 'Object', ha='center', va='bottom', fontsize=10, color='blue')
                
                # Calculate theoretical image position
                object_distance = element_positions[lens_idx]
                if object_distance > 0:
                    try:
                        s_i = 1 / (1/focal_length - 1/object_distance)
                        image_pos = element_positions[lens_idx] + abs(s_i)
                        image_pos_mm = image_pos * 1000
                        
                        # Draw theoretical sensor/image position with size
                        if sensor_size_mm:
                            sensor_half_height = sensor_size_mm / 4  # Make it visible
                            ax.plot([image_pos_mm, image_pos_mm], [-sensor_half_height, sensor_half_height], 
                                   color='green', linewidth=6, alpha=0.8, label='Sensor')
                            ax.text(image_pos_mm, -sensor_half_height - 2, 
                                   f'Sensor\\n{image_pos_mm:.1f}mm\\n{sensor_size_mm:.1f}mm', 
                                   ha='center', va='top', fontsize=9, color='green')
                        else:
                            ax.axvline(x=image_pos_mm, color='green', linewidth=4, alpha=0.8, label='Image Plane')
                            ax.text(image_pos_mm, -18, f'Image\\n{image_pos_mm:.1f}mm', 
                                   ha='center', va='top', fontsize=9, color='green')
                    except:
                        pass  # Skip if calculation fails
        
        # Draw spacings if provided
        if element_spacings:
            for i in range(len(positions_mm) - 1):
                start_pos = positions_mm[i]
                end_pos = positions_mm[i + 1]
                spacing = element_spacings[i] * 1000  # Convert to mm
                
                # Draw spacing arrow
                ax.annotate('', xy=(end_pos, -25), xytext=(start_pos, -25),
                           arrowprops=dict(arrowstyle='<->', color='gray', alpha=0.7))
                ax.text((start_pos + end_pos) / 2, -27, f'{spacing:.1f}mm',
                       ha='center', va='top', fontsize=9, color='gray')
        
        # Draw light rays
        ray_heights = [-12, -6, 0, 6, 12]
        
        # Calculate ray range including object and sensor positions
        all_positions = list(positions_mm)
        if include_imaging:
            lens_elements = [(i, elem) for i, elem in enumerate(elements) if hasattr(elem, 'focal_length')]
            if lens_elements:
                all_positions.append(0)  # Object at z=0
                # Add theoretical image position if calculable
                lens_idx, lens_element = lens_elements[0]
                object_distance = element_positions[lens_idx]
                focal_length = lens_element.focal_length
                try:
                    s_i = 1 / (1/focal_length - 1/object_distance)
                    image_pos_mm = (element_positions[lens_idx] + abs(s_i)) * 1000
                    all_positions.append(image_pos_mm)
                except:
                    pass
        
        start_pos = min(all_positions) - 5
        end_pos = max(all_positions) + 5
        
        for h in ray_heights:
            ax.plot([start_pos, end_pos], [h, h], 'b-', alpha=0.2, linewidth=1)
        
        # Formatting
        ax.set_xlim(start_pos - 2, end_pos + 2)
        ax.set_ylim(-35, 25)
        ax.set_xlabel('Position (mm)', fontsize=12)
        ax.set_ylabel('Height (mm)', fontsize=12)
        ax.set_title('Optical System Layout', fontsize=14, weight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        
        # Always save visualization
        import os
        os.makedirs('results', exist_ok=True)
        
        if save_path == 'auto':
            # Generate automatic filename based on system elements
            n_elements = len(elements)
            element_types = '_'.join([elem.__class__.__name__ for elem in elements[:3]])
            if len(elements) > 3:
                element_types += f'_plus{len(elements)-3}more'
            save_path = f'results/optical_system_{element_types}_{n_elements}elem.png'
        elif save_path and save_path != 'auto':
            # Use provided path
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"System geometry saved to: {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
        
        return fig