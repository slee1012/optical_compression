# Optical Compression Simulation Framework

A PyTorch-based framework for simulating optical compression systems as an alternative to H.265/JPEG.

## Installation

```bash
pip install -r requirements.txt
pip install -e .  # For development
```

## Quick Start

```python
from optical_compression import SystemBuilder, SystemConfig

# Configure
config = SystemConfig(sensor_resolution=(256, 256))

# Build system
system = SystemBuilder(config)\
    .add_phase_mask(position=0)\
    .add_coded_aperture(position=5e-3)\
    .build()

# Simulate
output = system(test_image)
```

## Examples

Run the examples:
```bash
python examples/basic_example.py
python examples/quick_demo.py
```

## Features

- Physical optics simulation with multiple propagation methods (angular spectrum, Fresnel, Fraunhofer)
- Modular optical elements (phase masks, coded apertures, lens arrays)
- End-to-end optimization with learnable elements
- Neural network decoders for reconstruction
- Comprehensive metrics and visualization tools

## License

MIT
