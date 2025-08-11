from dataclasses import dataclass, field
from typing import Tuple, Optional, Dict, Any
from enum import Enum
import numpy as np


class OpticalElementType(Enum):
    PHASE_MASK = "phase_mask"
    CODED_APERTURE = "coded_aperture"
    LENS_ARRAY = "lens_array"
    WAVEGUIDE = "waveguide"


class PropagationMethod(Enum):
    ANGULAR_SPECTRUM = "angular_spectrum"
    FRESNEL = "fresnel"
    FRAUNHOFER = "fraunhofer"


@dataclass
class SystemConfig:
    sensor_resolution: Tuple[int, int] = (256, 256)
    pixel_pitch: float = 5e-6
    bit_depth: int = 8
    readout_noise: float = 1.0
    wavelength: float = 550e-9
    wavelengths_rgb: Tuple[float, float, float] = (640e-9, 550e-9, 450e-9)
    focal_length: float = 50e-3
    f_number: float = 2.8
    propagation_distance: float = 10e-3
    propagation_method: PropagationMethod = PropagationMethod.ANGULAR_SPECTRUM
    oversample_factor: int = 2
    use_gpu: bool = True
    dtype: str = "float32"
    target_compression_ratio: float = 10.0
    subsample_factor: int = 1
    max_phase_modulation: float = 2 * np.pi
    temperature: float = 295.0
    learning_rate: float = 1e-3
    batch_size: int = 16
    num_epochs: int = 100
    
    @property
    def aperture_diameter(self):
        return self.focal_length / self.f_number
    
    @property
    def k(self):
        return 2 * np.pi / self.wavelength
    
    @property
    def fresnel_number(self):
        a = min(self.sensor_resolution) * self.pixel_pitch / 2
        return a**2 / (self.wavelength * self.propagation_distance)
    
    @property
    def nyquist_frequency(self):
        return 1 / (2 * self.pixel_pitch) * 1e-3
    
    @property
    def diffraction_limit(self):
        return 1.22 * self.wavelength * self.f_number
    
    def validate(self):
        checks = []
        sampling_ok = self.pixel_pitch < self.diffraction_limit / 2
        checks.append(("Sampling criterion", sampling_ok))
        fresnel_ok = self.fresnel_number > 0.1
        checks.append(("Fresnel approximation", fresnel_ok))
        sensor_size_ok = all(s > 0 for s in self.sensor_resolution)
        checks.append(("Sensor size", sensor_size_ok))
        
        print("Configuration Validation:")
        for name, status in checks:
            status_str = "✓" if status else "✗"
            print(f"  {status_str} {name}")
        print(f"  Fresnel number: {self.fresnel_number:.2f}")
        
        return all(status for _, status in checks)
    
    def to_dict(self):
        return {
            'sensor_resolution': self.sensor_resolution,
            'pixel_pitch': self.pixel_pitch,
            'wavelength': self.wavelength,
            'focal_length': self.focal_length,
            'f_number': self.f_number,
            'propagation_distance': self.propagation_distance,
            'oversample_factor': self.oversample_factor,
            'target_compression_ratio': self.target_compression_ratio
        }


@dataclass
class OpticalElementConfig:
    element_type: OpticalElementType
    position: float = 0.0
    parameters: Dict[str, Any] = field(default_factory=dict)
    learnable: bool = False
    name: Optional[str] = None
    
    def __post_init__(self):
        if self.name is None:
            self.name = f"{self.element_type.value}_{int(self.position*1e3)}mm"