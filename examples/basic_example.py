import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from config import SystemConfig
from system import SystemBuilder
from decoder import OpticalDecoder
from core import CompressionMetrics
from utils import Visualizer


def create_test_image(size=(256, 256), pattern='mixed'):
    H, W = size
    
    if pattern == 'checkerboard':
        block_size = 16
        pattern = torch.zeros(H, W)
        for i in range(0, H, block_size * 2):
            for j in range(0, W, block_size * 2):
                pattern[i:i+block_size, j:j+block_size] = 1
                pattern[i+block_size:i+2*block_size, j+block_size:j+2*block_size] = 1
    
    elif pattern == 'sinusoid':
        x = torch.linspace(0, 4*np.pi, W)
        y = torch.linspace(0, 4*np.pi, H)
        X, Y = torch.meshgrid(x, y, indexing='xy')
        pattern = 0.5 * (1 + torch.sin(X) * torch.cos(Y))
    
    else:  # mixed
        x = torch.linspace(-1, 1, W)
        y = torch.linspace(-1, 1, H)
        X, Y = torch.meshgrid(x, y, indexing='xy')
        
        pattern = 0.5 * (1 + torch.sin(4*np.pi*X) * torch.cos(4*np.pi*Y))
        
        for _ in range(5):
            cx, cy = torch.rand(2) * 2 - 1
            sigma = 0.1
            gaussian = torch.exp(-((X-cx)**2 + (Y-cy)**2)/(2*sigma**2))
            pattern += 0.3 * gaussian
        
        pattern = (pattern - pattern.min()) / (pattern.max() - pattern.min())
    
    return pattern.unsqueeze(0)


def main():
    print("=" * 60)
    print("Optical Compression Simulation - Basic Example")
    print("=" * 60)
    
    # 1. Configuration
    print("\n1. Setting up configuration...")
    
    config = SystemConfig(
        sensor_resolution=(256, 256),
        pixel_pitch=5e-6,
        wavelength=550e-9,
        focal_length=50e-3,
        f_number=2.8,
        propagation_distance=10e-3,
        target_compression_ratio=10.0,
        subsample_factor=2
    )
    
    print("\nValidating configuration:")
    config.validate()
    
    # 2. Build Optical System
    print("\n2. Building optical system...")
    
    system = SystemBuilder(config)        .add_phase_mask(position=0, learnable=True, init_type='random')        .add_coded_aperture(position=5e-3, aperture_type='binary', learnable=False)        .build()
    
    print(f"  - Added phase mask at 0 mm")
    print(f"  - Added coded aperture at 5 mm")
    
    # 3. Create Test Data
    print("\n3. Creating test images...")
    
    test_images = {
        'checkerboard': create_test_image(config.sensor_resolution, 'checkerboard'),
        'sinusoid': create_test_image(config.sensor_resolution, 'sinusoid'),
        'mixed': create_test_image(config.sensor_resolution, 'mixed')
    }
    
    # 4. Run Simulation
    print("\n4. Running optical simulation...")
    
    system.set_cache_intermediates(True)
    
    results = {}
    for name, image in test_images.items():
        print(f"\n  Processing {name} pattern...")
        
        output = system(image)
        
        results[name] = {
            'original': image,
            'sensor': output['intensity_sensor'],
            'field': output['field_sensor'],
            'intermediate': output['intermediate_fields'],
            'snr': output['snr']
        }
        
        print(f"    - SNR: {output['snr'].item():.2f} dB")
    
    # 5. Decode (Reconstruction)
    print("\n5. Reconstructing images...")
    
    decoder = OpticalDecoder(
        input_channels=1,
        output_channels=1,
        features=(32, 64, 128)
    )
    
    for name in results.keys():
        sensor_data = results[name]['sensor']
        
        with torch.no_grad():
            reconstructed = decoder(sensor_data)
        
        results[name]['reconstructed'] = reconstructed
    
    # 6. Calculate Metrics
    print("\n6. Calculating metrics...")
    
    metrics_calc = CompressionMetrics()
    
    for name in results.keys():
        original = results[name]['original'][0]
        sensor = results[name]['sensor'][0] if results[name]['sensor'].dim() > 2 else results[name]['sensor']
        reconstructed = results[name]['reconstructed'][0] if results[name]['reconstructed'].dim() > 2 else results[name]['reconstructed']
        
        metrics = {
            'mse': metrics_calc.mse(original, reconstructed),
            'psnr': metrics_calc.psnr(original, reconstructed),
            'ssim': metrics_calc.ssim(original, reconstructed),
            'compression_ratio': metrics_calc.compression_ratio(
                original.shape, 
                (sensor.shape[0]//config.subsample_factor, 
                 sensor.shape[1]//config.subsample_factor)
            ),
            'frequency_error': metrics_calc.frequency_error(original, reconstructed)
        }
        
        results[name]['metrics'] = metrics
        
        print(f"\n  {name.capitalize()} pattern:")
        print(f"    - PSNR: {metrics['psnr']:.2f} dB")
        print(f"    - SSIM: {metrics['ssim']:.4f}")
        print(f"    - Compression ratio: {metrics['compression_ratio']:.1f}x")
    
    # 7. Visualization
    print("\n7. Generating visualizations...")
    
    viz = Visualizer()
    
    # Plot pipeline for mixed pattern
    mixed_result = results['mixed']
    viz.plot_compression_pipeline(
        mixed_result['original'][0],
        mixed_result['sensor'][0] if mixed_result['sensor'].dim() > 2 else mixed_result['sensor'],
        mixed_result['reconstructed'][0] if mixed_result['reconstructed'].dim() > 2 else mixed_result['reconstructed'],
        mixed_result['intermediate']
    )
    
    # 8. Summary
    print("\n" + "=" * 60)
    print("Simulation Complete!")
    print("=" * 60)
    
    avg_psnr = np.mean([r['metrics']['psnr'] for r in results.values()])
    avg_ssim = np.mean([r['metrics']['ssim'] for r in results.values()])
    
    print(f"\nAverage Performance:")
    print(f"  - PSNR: {avg_psnr:.2f} dB")
    print(f"  - SSIM: {avg_ssim:.4f}")
    print(f"  - Target compression: {config.target_compression_ratio:.1f}x")
    
    print("\nPotential advantages over H.265/JPEG:")
    print("  - Optical preprocessing reduces sensor readout")
    print("  - Lower power consumption for edge devices")
    print("  - Inherent privacy through optical encoding")
    print("  - Task-specific optimization possible")
    
    return system, results


if __name__ == "__main__":
    system, results = main()