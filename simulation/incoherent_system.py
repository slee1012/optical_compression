"""
Incoherent (spectral) imaging simulation.
Merges spectral light source functionality with incoherent simulation.
"""

import torch
import numpy as np
from typing import Optional, Tuple, Dict, Any
from physics.light_sources import SpectralSource, SpectrumType


class IncoherentImagingSystem:
    """
    Incoherent spectral imaging system.
    
    Wraps a coherent system to perform spectral simulation by:
    1. Sampling wavelengths from a spectral source
    2. Running coherent simulation at each wavelength
    3. Incoherently summing intensity results
    """
    
    def __init__(self, 
                 coherent_system,
                 spectral_source: Optional[SpectralSource] = None,
                 n_wavelength_samples: int = 5):
        """
        Initialize incoherent imaging system.
        
        Args:
            coherent_system: Built coherent imaging system
            spectral_source: Light source spectral definition
            n_wavelength_samples: Number of wavelengths to sample
        """
        self.coherent_system = coherent_system
        self.n_samples = n_wavelength_samples
        
        # Default to daylight if no source specified
        if spectral_source is None:
            spectral_source = SpectralSource(spectrum_type=SpectrumType.DAYLIGHT)
        
        self.spectral_source = spectral_source
        
        # Sample wavelengths for simulation
        self.wavelengths, self.spectral_weights = self.spectral_source.sample_wavelengths(n_wavelength_samples)
        
        # Store original system wavelength
        self.original_wavelength = self.coherent_system.config.wavelength
        
    def __call__(self, input_field: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Run incoherent imaging simulation.
        
        Args:
            input_field: Input field or intensity pattern
            
        Returns:
            Dictionary containing simulation results:
            - 'intensity_sensor': Incoherent intensity at sensor
            - 'wavelength_results': Per-wavelength results
            - 'spectral_weights': Wavelength weights used
        """
        device = input_field.device
        
        # Initialize accumulated intensity
        sensor_shape = input_field.shape
        accumulated_intensity = torch.zeros(sensor_shape, dtype=torch.float32, device=device)
        wavelength_results = []
        
        # Iterate over sampled wavelengths
        for i, (wavelength, weight) in enumerate(zip(self.wavelengths, self.spectral_weights)):
            # Update system wavelength
            self.coherent_system.config.wavelength = wavelength
            
            # Update all optical elements for new wavelength if needed
            self._update_elements_wavelength(wavelength)
            
            # Convert input to complex field if needed
            if torch.is_complex(input_field):
                field = input_field
            else:
                # For incoherent simulation, input might be intensity
                field = torch.complex(torch.sqrt(input_field.clamp(min=0)), 
                                    torch.zeros_like(input_field))
            
            # Run coherent simulation
            output = self.coherent_system(field)
            
            # Extract intensity
            if isinstance(output, dict):
                if 'intensity_sensor' in output:
                    intensity = output['intensity_sensor']
                elif 'field_sensor' in output:
                    intensity = torch.abs(output['field_sensor'])**2
                else:
                    # Fallback to first tensor value
                    intensity = torch.abs(list(output.values())[0])**2
            else:
                intensity = torch.abs(output)**2
            
            # Ensure compatible shapes for accumulation
            if intensity.shape != accumulated_intensity.shape:
                if intensity.ndim == accumulated_intensity.ndim + 1:
                    intensity = intensity.squeeze(0)  # Remove batch dimension
                elif accumulated_intensity.ndim == intensity.ndim + 1:
                    accumulated_intensity = accumulated_intensity.squeeze(0)
            
            # Accumulate with spectral weight
            accumulated_intensity += weight * intensity
            
            # Store per-wavelength result
            wavelength_results.append({
                'wavelength': wavelength,
                'weight': weight,
                'intensity': intensity.detach().clone()
            })
        
        # Restore original wavelength
        self.coherent_system.config.wavelength = self.original_wavelength
        self._update_elements_wavelength(self.original_wavelength)
        
        return {
            'intensity_sensor': accumulated_intensity,
            'wavelength_results': wavelength_results,
            'spectral_weights': self.spectral_weights,
            'wavelengths': self.wavelengths
        }
    
    def simulate_rgb(self, input_field: torch.Tensor) -> torch.Tensor:
        """
        Simulate RGB color imaging.
        
        Args:
            input_field: Input field pattern
            
        Returns:
            RGB tensor with shape [3, H, W] for R, G, B channels
        """
        rgb_wavelengths = self.spectral_source.get_rgb_wavelengths()  # 640nm, 550nm, 450nm
        rgb_results = []
        
        for wavelength in rgb_wavelengths:
            # Update system wavelength
            self.coherent_system.config.wavelength = wavelength
            self._update_elements_wavelength(wavelength)
            
            # Convert input to complex field if needed
            if torch.is_complex(input_field):
                field = input_field
            else:
                field = torch.complex(torch.sqrt(input_field.clamp(min=0)), 
                                    torch.zeros_like(input_field))
            
            # Run coherent simulation
            output = self.coherent_system(field)
            
            # Extract intensity
            if isinstance(output, dict):
                if 'intensity_sensor' in output:
                    intensity = output['intensity_sensor']
                elif 'field_sensor' in output:
                    intensity = torch.abs(output['field_sensor'])**2
                else:
                    intensity = torch.abs(list(output.values())[0])**2
            else:
                intensity = torch.abs(output)**2
            
            # Remove batch dimension if present
            if intensity.ndim == 3 and intensity.shape[0] == 1:
                intensity = intensity.squeeze(0)
            
            rgb_results.append(intensity)
        
        # Restore original wavelength
        self.coherent_system.config.wavelength = self.original_wavelength
        self._update_elements_wavelength(self.original_wavelength)
        
        return torch.stack(rgb_results, dim=0)  # Shape: [3, H, W]
    
    def _update_elements_wavelength(self, wavelength: float):
        """Update wavelength in all optical elements."""
        if hasattr(self.coherent_system, 'elements'):
            for element in self.coherent_system.elements:
                if hasattr(element, 'wavelength'):
                    element.wavelength = wavelength
                elif hasattr(element, 'config') and hasattr(element.config, 'wavelength'):
                    element.config.wavelength = wavelength
    
    def set_spectral_source(self, spectral_source: SpectralSource, n_samples: Optional[int] = None):
        """Change the spectral source and resample wavelengths."""
        self.spectral_source = spectral_source
        if n_samples is not None:
            self.n_samples = n_samples
        
        # Resample wavelengths
        self.wavelengths, self.spectral_weights = spectral_source.sample_wavelengths(self.n_samples)
    
    def get_spectral_info(self) -> Dict[str, Any]:
        """Get information about the spectral simulation setup."""
        return {
            'spectrum_type': self.spectral_source.spectrum_type,
            'n_samples': self.n_samples,
            'wavelengths': self.wavelengths,
            'spectral_weights': self.spectral_weights,
            'wavelength_range': (self.wavelengths.min(), self.wavelengths.max()),
            'original_wavelength': self.original_wavelength
        }


# Convenient factory functions
def create_daylight_system(coherent_system, n_samples: int = 7) -> IncoherentImagingSystem:
    """Create incoherent system with daylight illumination."""
    source = SpectralSource(spectrum_type=SpectrumType.DAYLIGHT)
    return IncoherentImagingSystem(coherent_system, source, n_samples)

def create_led_system(coherent_system, peak_wavelength: float = 550e-9, 
                     bandwidth: float = 50e-9, n_samples: int = 5) -> IncoherentImagingSystem:
    """Create incoherent system with LED illumination."""
    source = SpectralSource(
        spectrum_type=SpectrumType.LED,
        peak_wavelength=peak_wavelength,
        bandwidth=bandwidth
    )
    return IncoherentImagingSystem(coherent_system, source, n_samples)

def create_blackbody_system(coherent_system, temperature: float = 5500, 
                           n_samples: int = 7) -> IncoherentImagingSystem:
    """Create incoherent system with blackbody illumination."""
    source = SpectralSource(
        spectrum_type=SpectrumType.BLACKBODY,
        temperature=temperature
    )
    return IncoherentImagingSystem(coherent_system, source, n_samples)