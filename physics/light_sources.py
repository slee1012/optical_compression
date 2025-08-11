"""
Spectral light source definitions for incoherent imaging simulation.
"""

import torch
import numpy as np
from dataclasses import dataclass
from typing import Tuple, List, Optional, Union
from enum import Enum


class SpectrumType(Enum):
    MONOCHROMATIC = "monochromatic"
    BLACKBODY = "blackbody"
    LED = "led"
    FLUORESCENT = "fluorescent"
    DAYLIGHT = "daylight"
    CUSTOM = "custom"


@dataclass
class SpectralSource:
    """Defines spectral characteristics of light sources."""
    
    spectrum_type: SpectrumType = SpectrumType.DAYLIGHT
    wavelengths: Optional[np.ndarray] = None  # in meters
    intensities: Optional[np.ndarray] = None  # normalized spectral power
    temperature: float = 5500.0  # Color temperature in Kelvin (for blackbody)
    peak_wavelength: float = 550e-9  # Peak wavelength for LED (in meters)
    bandwidth: float = 50e-9  # FWHM bandwidth for LED (in meters)
    
    def __post_init__(self):
        """Generate spectrum based on type if not provided."""
        if self.wavelengths is None:
            # Default wavelength range: 400-700nm with 10nm spacing
            self.wavelengths = np.linspace(400e-9, 700e-9, 31)
        
        if self.intensities is None:
            self.intensities = self._generate_spectrum()
    
    def _generate_spectrum(self) -> np.ndarray:
        """Generate spectral intensity distribution based on source type."""
        if self.spectrum_type == SpectrumType.MONOCHROMATIC:
            # Single wavelength
            intensities = np.zeros_like(self.wavelengths)
            idx = np.argmin(np.abs(self.wavelengths - self.peak_wavelength))
            intensities[idx] = 1.0
            
        elif self.spectrum_type == SpectrumType.BLACKBODY:
            # Planck's law
            intensities = self._planck_spectrum(self.wavelengths, self.temperature)
            
        elif self.spectrum_type == SpectrumType.LED:
            # Gaussian-like spectrum
            intensities = np.exp(-((self.wavelengths - self.peak_wavelength) / (self.bandwidth/2.355))**2)
            
        elif self.spectrum_type == SpectrumType.DAYLIGHT:
            # CIE D65 approximation
            intensities = self._d65_spectrum(self.wavelengths)
            
        elif self.spectrum_type == SpectrumType.FLUORESCENT:
            # Simplified fluorescent with peaks
            intensities = self._fluorescent_spectrum(self.wavelengths)
            
        else:
            # Default uniform spectrum
            intensities = np.ones_like(self.wavelengths)
        
        # Normalize
        intensities = intensities / np.max(intensities)
        return intensities
    
    def _planck_spectrum(self, wavelengths: np.ndarray, temperature: float) -> np.ndarray:
        """Calculate Planck's blackbody radiation spectrum."""
        h = 6.626e-34  # Planck's constant
        c = 3e8  # Speed of light
        k = 1.381e-23  # Boltzmann constant
        
        # Planck's law
        with np.errstate(over='ignore', invalid='ignore'):
            intensity = (2 * h * c**2 / wavelengths**5) / \
                       (np.exp(h * c / (wavelengths * k * temperature)) - 1)
            intensity = np.nan_to_num(intensity, 0)
        
        return intensity
    
    def _d65_spectrum(self, wavelengths: np.ndarray) -> np.ndarray:
        """Approximate CIE D65 daylight spectrum."""
        # Simplified D65 using Gaussian components
        wavelengths_nm = wavelengths * 1e9
        
        # Three main peaks in D65
        intensity = (
            0.8 * np.exp(-((wavelengths_nm - 460) / 60)**2) +  # Blue peak
            1.0 * np.exp(-((wavelengths_nm - 550) / 80)**2) +  # Green peak
            0.9 * np.exp(-((wavelengths_nm - 610) / 70)**2)    # Red peak
        )
        
        return intensity
    
    def _fluorescent_spectrum(self, wavelengths: np.ndarray) -> np.ndarray:
        """Simplified fluorescent lamp spectrum with mercury lines."""
        wavelengths_nm = wavelengths * 1e9
        
        # Mercury emission lines plus phosphor continuum
        intensity = 0.3 * np.ones_like(wavelengths)  # Phosphor background
        
        # Add mercury peaks
        mercury_lines = [435.8, 546.1, 577.0, 579.0]  # nm
        for line in mercury_lines:
            intensity += 0.5 * np.exp(-((wavelengths_nm - line) / 2)**2)
        
        return intensity
    
    def sample_wavelengths(self, n_samples: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        """
        Sample wavelengths from the spectrum for simulation.
        
        Args:
            n_samples: Number of wavelengths to sample
            
        Returns:
            sampled_wavelengths: Array of sampled wavelengths
            weights: Intensity weights for each wavelength
        """
        if n_samples == 1:
            # For monochromatic, use peak wavelength
            idx = np.argmax(self.intensities)
            return np.array([self.wavelengths[idx]]), np.array([1.0])
        
        # Sample based on intensity distribution
        # Use importance sampling
        cumsum = np.cumsum(self.intensities)
        cumsum = cumsum / cumsum[-1]
        
        # Uniform sampling in CDF space
        samples = np.linspace(0.05, 0.95, n_samples)
        indices = np.searchsorted(cumsum, samples)
        
        sampled_wavelengths = self.wavelengths[indices]
        weights = self.intensities[indices]
        
        # Normalize weights
        weights = weights / np.sum(weights)
        
        return sampled_wavelengths, weights
    
    def get_rgb_wavelengths(self) -> Tuple[float, float, float]:
        """Get standard RGB wavelengths for color imaging."""
        return (640e-9, 550e-9, 450e-9)  # Red, Green, Blue
    
    def to_torch(self, device: Optional[torch.device] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Convert spectrum to PyTorch tensors."""
        wavelengths = torch.tensor(self.wavelengths, dtype=torch.float32, device=device)
        intensities = torch.tensor(self.intensities, dtype=torch.float32, device=device)
        return wavelengths, intensities


# Preset light sources
SOURCES = {
    'daylight': SpectralSource(spectrum_type=SpectrumType.DAYLIGHT),
    'tungsten': SpectralSource(spectrum_type=SpectrumType.BLACKBODY, temperature=3200),
    'led_white': SpectralSource(spectrum_type=SpectrumType.LED, peak_wavelength=550e-9, bandwidth=100e-9),
    'led_blue': SpectralSource(spectrum_type=SpectrumType.LED, peak_wavelength=450e-9, bandwidth=30e-9),
    'fluorescent': SpectralSource(spectrum_type=SpectrumType.FLUORESCENT),
    'laser_green': SpectralSource(spectrum_type=SpectrumType.MONOCHROMATIC, peak_wavelength=532e-9),
}