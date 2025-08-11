"""
Lens optical elements - thin lenses, thick lenses, and lens systems.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple, Dict, Any
from .base import OpticalElement


class ThinLens(OpticalElement):
    """
    Thin lens element implementing quadratic phase transformation.
    
    Applies phase shift: exp(-i * k * r^2 / (2*f))
    where f is focal length, r is radial distance, k is wavenumber.
    
    Also applies circular aperture limiting if specified.
    """
    
    def __init__(self, 
                 resolution: Tuple[int, int],
                 pixel_pitch: float,
                 wavelength: float,
                 focal_length: float,
                 aperture_diameter: Optional[float] = None,
                 name: Optional[str] = None):
        """
        Initialize thin lens element.
        
        Args:
            resolution: (height, width) in pixels
            pixel_pitch: Physical size of each pixel (meters)
            wavelength: Operating wavelength (meters)
            focal_length: Lens focal length (meters)
            aperture_diameter: Clear aperture diameter (meters), None for no limit
            name: Optional name for the element
        """
        self.focal_length = focal_length
        self.aperture_diameter = aperture_diameter
        
        super().__init__(resolution, pixel_pitch, wavelength, name)
        
    def _init_parameters(self):
        """Initialize lens-specific parameters."""
        # Create coordinate grids
        H, W = self.resolution
        y = torch.arange(H, dtype=torch.float32) - H/2
        x = torch.arange(W, dtype=torch.float32) - W/2
        Y, X = torch.meshgrid(y, x, indexing='ij')
        
        # Convert to physical coordinates  
        Y = Y * self.pixel_pitch
        X = X * self.pixel_pitch
        R_squared = X**2 + Y**2
        
        # Register as buffers (non-trainable parameters)
        self.register_buffer('Y', Y)
        self.register_buffer('X', X) 
        self.register_buffer('R_squared', R_squared)
        
        # Precompute aperture mask if specified
        if self.aperture_diameter is not None:
            aperture_radius = self.aperture_diameter / 2
            aperture_mask = (torch.sqrt(R_squared) <= aperture_radius).float()
            self.register_buffer('aperture_mask', aperture_mask)
        else:
            self.register_buffer('aperture_mask', torch.ones_like(R_squared))
            
    def forward(self, field: torch.Tensor) -> torch.Tensor:
        """
        Apply thin lens transformation.
        
        Args:
            field: Input complex field tensor
            
        Returns:
            Output field after lens transformation
        """
        field = self.ensure_complex(field)
        
        # Calculate lens phase (wavelength dependent!)
        k = 2 * np.pi / self.wavelength
        lens_phase = -k * self.R_squared / (2 * self.focal_length)
        
        # Apply lens transformation with aperture
        lens_response = self.aperture_mask * torch.exp(1j * lens_phase)
        
        return field * lens_response
    
    def update_wavelength(self, wavelength: float):
        """
        Update wavelength for spectral simulations.
        
        Args:
            wavelength: New wavelength (meters)
        """
        self.wavelength = wavelength
        self.k = 2 * np.pi / wavelength
    
    def get_params_dict(self) -> Dict[str, Any]:
        """Get lens parameters as dictionary."""
        params = super().get_params_dict()
        params.update({
            'focal_length': self.focal_length,
            'aperture_diameter': self.aperture_diameter,
            'f_number': self.get_f_number(),
            'numerical_aperture': self.get_numerical_aperture()
        })
        return params
    
    def get_f_number(self) -> Optional[float]:
        """Calculate f-number (f/#) of the lens."""
        if self.aperture_diameter is not None:
            return self.focal_length / self.aperture_diameter
        return None
    
    def get_numerical_aperture(self) -> Optional[float]:
        """Calculate numerical aperture (NA) of the lens."""
        if self.aperture_diameter is not None:
            # NA = sin(theta) where theta is half-angle of cone
            # For thin lens: sin(theta) ≈ D/(2*f) for small angles
            return self.aperture_diameter / (2 * self.focal_length)
        return None
    
    def visualize(self) -> Dict[str, torch.Tensor]:
        """Return visualization data for the lens."""
        # Lens phase pattern
        k = 2 * np.pi / self.wavelength  
        lens_phase = -k * self.R_squared / (2 * self.focal_length)
        
        # Wrapped phase for visualization
        phase_wrapped = torch.remainder(lens_phase + np.pi, 2*np.pi) - np.pi
        
        return {
            'lens_phase': lens_phase,
            'phase_wrapped': phase_wrapped,
            'aperture_mask': self.aperture_mask,
            'coordinate_x': self.X,
            'coordinate_y': self.Y,
            'radial_distance': torch.sqrt(self.R_squared)
        }


class ThickLens(OpticalElement):
    """
    Thick lens element with more realistic modeling.
    
    Models lens with finite thickness and curved surfaces.
    Uses paraxial approximation for phase calculation.
    """
    
    def __init__(self,
                 resolution: Tuple[int, int], 
                 pixel_pitch: float,
                 wavelength: float,
                 focal_length: float,
                 thickness: float,
                 front_radius: float,
                 back_radius: float,
                 refractive_index: float = 1.5,
                 aperture_diameter: Optional[float] = None,
                 name: Optional[str] = None):
        """
        Initialize thick lens element.
        
        Args:
            resolution: (height, width) in pixels
            pixel_pitch: Physical pixel size (meters)
            wavelength: Operating wavelength (meters) 
            focal_length: Effective focal length (meters)
            thickness: Lens thickness at center (meters)
            front_radius: Front surface radius of curvature (meters)
            back_radius: Back surface radius of curvature (meters)  
            refractive_index: Lens material refractive index
            aperture_diameter: Clear aperture diameter (meters)
            name: Optional element name
        """
        self.focal_length = focal_length
        self.thickness = thickness
        self.front_radius = front_radius
        self.back_radius = back_radius
        self.refractive_index = refractive_index
        self.aperture_diameter = aperture_diameter
        
        super().__init__(resolution, pixel_pitch, wavelength, name)
    
    def _init_parameters(self):
        """Initialize thick lens parameters."""
        # Create coordinate grids
        H, W = self.resolution
        y = torch.arange(H, dtype=torch.float32) - H/2
        x = torch.arange(W, dtype=torch.float32) - W/2
        Y, X = torch.meshgrid(y, x, indexing='ij')
        
        # Convert to physical coordinates
        Y = Y * self.pixel_pitch
        X = X * self.pixel_pitch
        R_squared = X**2 + Y**2
        
        self.register_buffer('Y', Y)
        self.register_buffer('X', X)
        self.register_buffer('R_squared', R_squared)
        
        # Aperture mask
        if self.aperture_diameter is not None:
            aperture_radius = self.aperture_diameter / 2
            aperture_mask = (torch.sqrt(R_squared) <= aperture_radius).float()
            self.register_buffer('aperture_mask', aperture_mask)
        else:
            self.register_buffer('aperture_mask', torch.ones_like(R_squared))
    
    def forward(self, field: torch.Tensor) -> torch.Tensor:
        """
        Apply thick lens transformation.
        
        Uses paraxial approximation for phase calculation including
        both surface contributions and path length difference.
        """
        field = self.ensure_complex(field)
        
        # Calculate phase contributions
        k = 2 * np.pi / self.wavelength
        k_lens = 2 * np.pi * self.refractive_index / self.wavelength
        
        # Front surface contribution
        if abs(self.front_radius) > 1e-10:  # Avoid division by zero
            front_phase = -k * self.R_squared / (2 * self.front_radius)
        else:
            front_phase = torch.zeros_like(self.R_squared)
        
        # Back surface contribution  
        if abs(self.back_radius) > 1e-10:
            back_phase = k * self.R_squared / (2 * self.back_radius)
        else:
            back_phase = torch.zeros_like(self.R_squared)
        
        # Path length difference through lens
        # For paraxial approximation: extra path ≈ r²/(2R) for each surface
        path_difference = (self.R_squared / (2 * abs(self.front_radius)) + 
                          self.R_squared / (2 * abs(self.back_radius)))
        path_phase = (k_lens - k) * path_difference
        
        # Total lens phase
        total_phase = front_phase + back_phase + path_phase
        
        # Apply transformation
        lens_response = self.aperture_mask * torch.exp(1j * total_phase)
        
        return field * lens_response
    
    def update_wavelength(self, wavelength: float):
        """Update wavelength for thick lens."""
        self.wavelength = wavelength
        self.k = 2 * np.pi / wavelength
    
    def get_params_dict(self) -> Dict[str, Any]:
        """Get thick lens parameters."""
        params = super().get_params_dict()
        params.update({
            'focal_length': self.focal_length,
            'thickness': self.thickness,
            'front_radius': self.front_radius, 
            'back_radius': self.back_radius,
            'refractive_index': self.refractive_index,
            'aperture_diameter': self.aperture_diameter
        })
        return params


# Convenience factory functions
def create_thin_lens(focal_length: float,
                    aperture_diameter: float,
                    resolution: Tuple[int, int] = (256, 256),
                    pixel_pitch: float = 5e-6,
                    wavelength: float = 550e-9,
                    name: Optional[str] = None) -> ThinLens:
    """
    Create a thin lens element with common parameters.
    
    Args:
        focal_length: Lens focal length (meters)
        aperture_diameter: Clear aperture diameter (meters)
        resolution: Sensor resolution (pixels)
        pixel_pitch: Pixel size (meters)
        wavelength: Operating wavelength (meters)
        name: Optional element name
        
    Returns:
        ThinLens element
    """
    return ThinLens(
        resolution=resolution,
        pixel_pitch=pixel_pitch, 
        wavelength=wavelength,
        focal_length=focal_length,
        aperture_diameter=aperture_diameter,
        name=name
    )

def create_achromatic_doublet(focal_length: float,
                             aperture_diameter: float,
                             crown_index: float = 1.517,
                             flint_index: float = 1.620,
                             resolution: Tuple[int, int] = (256, 256),
                             pixel_pitch: float = 5e-6,
                             wavelength: float = 550e-9,
                             name: Optional[str] = None) -> nn.Module:
    """
    Create an achromatic doublet approximation using two thin lenses.
    
    Simple model for reduced chromatic aberration.
    """
    # Simplified doublet: positive crown + negative flint
    # Powers calculated to minimize chromatic aberration
    total_power = 1.0 / focal_length
    
    # Approximate power distribution for achromatic doublet
    crown_power = total_power * 1.2  # Positive element (stronger)
    flint_power = total_power * -0.2  # Negative element (weaker)
    
    crown_focal = 1.0 / crown_power if abs(crown_power) > 1e-10 else 1e6
    flint_focal = 1.0 / flint_power if abs(flint_power) > 1e-10 else 1e6
    
    class AchromaticDoublet(nn.Module):
        def __init__(self):
            super().__init__()
            self.crown = ThinLens(resolution, pixel_pitch, wavelength, 
                                crown_focal, aperture_diameter, 
                                name=f"{name}_crown" if name else "crown")
            self.flint = ThinLens(resolution, pixel_pitch, wavelength,
                                flint_focal, aperture_diameter,
                                name=f"{name}_flint" if name else "flint")
            
        def forward(self, field):
            field = self.crown(field)
            field = self.flint(field)
            return field
            
        def update_wavelength(self, wavelength):
            self.crown.update_wavelength(wavelength)
            self.flint.update_wavelength(wavelength)
    
    return AchromaticDoublet()