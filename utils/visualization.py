import torch
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict, Optional


class Visualizer:
    
    def __init__(self, figsize=(15, 10)):
        self.figsize = figsize
    
    def plot_compression_pipeline(self, original, sensor_output, reconstructed=None, 
                                 intermediate_fields=None, save_path=None):
        
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
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
    
    def plot_metrics(self, metrics_history):
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
        plt.show()
    
    def plot_optical_elements(self, elements):
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
        plt.show()