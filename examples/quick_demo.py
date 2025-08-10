import sys
import os

# Add parent directory to path properly
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import torch
import numpy as np
import matplotlib.pyplot as plt

# Use absolute imports
from optical_compression.config import SystemConfig
from optical_compression.system import SystemBuilder
from optical_compression.decoder import OpticalDecoder
from optical_compression.core import CompressionMetrics
from optical_compression.utils import Visualizer


def create_demo_image(size=(256, 256)):
    H, W = size
    x = torch.linspace(-1, 1, W)
    y = torch.linspace(-1, 1, H)
    X, Y = torch.meshgrid(x, y, indexing='xy')
    
    pattern = torch.zeros(H, W)
    
    # Add circles
    for i in range(3):
        cx, cy = torch.rand(2) * 1.6 - 0.8
        r = torch.rand(1) * 0.2 + 0.1
        circle = torch.exp(-((X - cx)**2 + (Y - cy)**2) / (2 * r**2))
        pattern += circle
    
    # Add gradient
    pattern += 0.3 * (X + Y) / 2
    
    # Normalize
    pattern = (pattern - pattern.min()) / (pattern.max() - pattern.min())
    
    return pattern.unsqueeze(0)


def main():
    print("=" * 60)
    print("Optical Compression - Quick Demo")
    print("=" * 60)
    
    # Setup
    print("\n1. Setting up optical system...")
    config = SystemConfig(
        sensor_resolution=(256, 256),
        pixel_pitch=5e-6,
        wavelength=550e-9,
        propagation_distance=10e-3
    )
    
    system = SystemBuilder(config)        .add_phase_mask(position=0, learnable=True)        .add_coded_aperture(position=5e-3, aperture_type='binary')        .build()
    
    # Create test image
    print("2. Creating test image...")
    test_image = create_demo_image(config.sensor_resolution)
    
    # Simulate
    print("3. Running optical simulation...")
    system.set_cache_intermediates(True)
    output = system(test_image)
    
    # Decode
    print("4. Applying decoder...")
    decoder = OpticalDecoder(input_channels=1, output_channels=1, features=(32, 64))
    
    with torch.no_grad():
        sensor_data = output['intensity_sensor']
        reconstructed = decoder(sensor_data)
    
    # Metrics
    print("5. Computing metrics...")
    metrics = CompressionMetrics()
    
    original = test_image[0]
    recon = reconstructed[0] if reconstructed.dim() > 2 else reconstructed
    
    psnr = metrics.psnr(original, recon)
    compression_ratio = metrics.compression_ratio(
        original.shape,
        (sensor_data[0].shape[0]//2, sensor_data[0].shape[1]//2) if sensor_data.dim() > 2 else (sensor_data.shape[0]//2, sensor_data.shape[1]//2)
    )
    
    print(f"\nResults:")
    print(f"  - PSNR: {psnr:.2f} dB")
    print(f"  - Compression Ratio: {compression_ratio:.1f}x")
    print(f"  - SNR at sensor: {output['snr'].item():.2f} dB")
    
    # Visualize
    print("\n6. Generating visualization...")
    viz = Visualizer()
    viz.plot_compression_pipeline(
        test_image[0],
        output['intensity_sensor'][0] if output['intensity_sensor'].dim() > 2 else output['intensity_sensor'],
        recon,
        output['intermediate_fields']
    )
    
    print("\n" + "=" * 60)
    print("Demo complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()