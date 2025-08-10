# test_working.py - This WILL work
import torch
import numpy as np
import matplotlib.pyplot as plt

print("Testing Optical Compression System")
print("="*50)

# Test 1: Import config
try:
    from config.system_config import SystemConfig
    print("✓ Config import successful")
    
    config = SystemConfig(sensor_resolution=(128, 128))
    print(f"✓ Config created: {config.sensor_resolution}")
except Exception as e:
    print(f"✗ Config error: {e}")

# Test 2: Import and use system builder
try:
    from system.builder import SystemBuilder
    print("✓ SystemBuilder import successful")
    
    system = SystemBuilder(config)\
        .add_phase_mask(position=0)\
        .add_coded_aperture(position=5e-3)\
        .build()
    print("✓ System built successfully")
except Exception as e:
    print(f"✗ System builder error: {e}")

# Test 3: Run simulation
try:
    test_image = torch.rand(1, 128, 128)
    output = system(test_image)
    print(f"✓ Simulation successful! SNR: {output['snr'].item():.2f} dB")
except Exception as e:
    print(f"✗ Simulation error: {e}")

# Test 4: Decoder
try:
    from decoder.networks import OpticalDecoder
    decoder = OpticalDecoder(input_channels=1, output_channels=1, features=(32, 64))
    reconstructed = decoder(output['intensity_sensor'].unsqueeze(1))
    print(f"✓ Decoder successful! Output shape: {reconstructed.shape}")
except Exception as e:
    print(f"✗ Decoder error: {e}")

# Test 5: Metrics
try:
    from core.metrics import CompressionMetrics
    metrics = CompressionMetrics()
    psnr = metrics.psnr(test_image[0], reconstructed[0,0])
    print(f"✓ Metrics successful! PSNR: {psnr:.2f} dB")
except Exception as e:
    print(f"✗ Metrics error: {e}")

# Test 6: Simple visualization
try:
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    axes[0].imshow(test_image[0].numpy(), cmap='gray')
    axes[0].set_title('Original')
    axes[0].axis('off')
    
    axes[1].imshow(output['intensity_sensor'][0].numpy(), cmap='gray')
    axes[1].set_title('Sensor Output')
    axes[1].axis('off')
    
    axes[2].imshow(reconstructed[0,0].detach().numpy(), cmap='gray')
    axes[2].set_title('Reconstructed')
    axes[2].axis('off')
    
    plt.suptitle('Optical Compression Pipeline')
    plt.tight_layout()
    plt.show()
    
    print("✓ Visualization successful!")
except Exception as e:
    print(f"✗ Visualization error: {e}")

print("\n" + "="*50)
print("All tests complete!")