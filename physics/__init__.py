"""
Physics module for optical wave simulation.

Contains fundamental physics operations:
- Wave propagation methods (Angular Spectrum, Fresnel, etc.)
- Light source modeling (spectral sources, coherence)
"""

from .propagation import OpticalPropagation
from .light_sources import SpectralSource, SpectrumType

__all__ = ['OpticalPropagation', 'SpectralSource', 'SpectrumType']