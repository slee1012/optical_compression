# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview
PyTorch-based optical compression framework simulating learnable optical elements as an alternative to H.265/JPEG compression. The system performs compression in the optical domain before digitization for power-efficient edge devices.

## Development Commands

### Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Install in development mode  
pip install -e .
```

### Running Code
```bash
# Run comprehensive demo
python examples/basic_example.py

# Quick functionality test
python examples/quick_demo.py  

# Integration tests
python test_working.py
```

### Testing
```bash
# Run unit tests (located in tests/)
python -m pytest tests/

# Run integration test
python test_working.py
```

## Core Architecture

### Module Organization
- **config/**: SystemConfig dataclass and presets - all system parameters flow through here
- **physics/**: Core physics operations (wave propagation, light sources)
- **simulation/**: System-level simulation (coherent/incoherent systems, builders)
- **elements/**: Optical components (lenses, apertures, phase masks - all inherit from OpticalElement)
- **analysis/**: Evaluation tools (metrics, visualization, plotting)

### Key Design Patterns
1. **Builder Pattern**: Use SystemBuilder for creating optical systems
   ```python
   system = SystemBuilder(config).add_phase_mask().add_coded_aperture().build()
   ```

2. **PyTorch Integration**: All optical elements are nn.Module subclasses with learnable nn.Parameters

3. **Complex Field Handling**: The entire pipeline operates on complex tensors for amplitude/phase

### Critical Implementation Details
- **Propagation Methods**: Angular Spectrum (default) or Fresnel in physics/propagation.py
- **Coordinate System**: Physical units (meters) with wavelength-dependent sampling
- **Gradient Flow**: All operations maintain differentiability for end-to-end learning
- **Spectral Simulation**: Incoherent systems handle wavelength-dependent effects

## Common Development Tasks

### Adding New Optical Elements
1. Create new class in elements/ inheriting from OpticalElement
2. Implement forward() method operating on complex fields
3. Register learnable parameters as nn.Parameter
4. Add to SystemBuilder methods

### Modifying System Configuration  
Edit config/system_config.py SystemConfig dataclass - includes validation and computed properties

### Running Experiments
Use config/presets.py for standard configurations or create new SystemConfig instances

## Important Files to Know
- **simulation/coherent_system.py**: Main coherent imaging pipeline
- **simulation/incoherent_system.py**: Spectral (incoherent) imaging simulation
- **physics/propagation.py**: Wave propagation physics (Angular Spectrum, Fresnel)
- **elements/base.py**: OpticalElement interface all components must implement
- **elements/lens.py**: Lens elements (thin, thick, achromatic doublets)

## Testing Strategy
- Unit tests for individual optical elements in tests/
- Integration test in test_working.py validates full pipeline
- Use .detach() when visualizing tensors to avoid gradient tracking issues