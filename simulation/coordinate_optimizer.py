"""
Coordinate system optimizer for optical simulations.

Analyzes optical system parameters and determines optimal:
- Resolution (sampling)
- Pixel pitch 
- Field size
- Padding requirements

To minimize numerical artifacts and ensure accurate propagation.
"""

import torch
import numpy as np
from typing import Tuple, Dict, List, Optional
from dataclasses import dataclass

@dataclass
class CoordinateConfig:
    """Optimized coordinate system configuration."""
    resolution: Tuple[int, int]
    pixel_pitch: float
    field_size: Tuple[float, float]  # Physical field size in meters
    padding_factor: float
    fresnel_number: float
    max_spatial_freq: float  # Maximum spatial frequency (1/m)
    recommended_sensor_size: Tuple[float, float]  # Recommended sensor physical size
    warnings: List[str]


class CoordinateOptimizer:
    """
    Optimizes coordinate system parameters for optical simulations.
    
    Logic Flow:
    1. Input resolution and pixel size define initial field size and max diffraction angle
    2. Calculate required padding for each propagation distance based on diffraction spreading
    3. Each aperture acts as a coordinate reset point - validate aperture size vs sampling limits
    4. Show warnings when apertures must be resized due to memory/sampling constraints
    """
    
    def __init__(self, wavelength: float, memory_limit_3k: bool = True):
        """
        Initialize optimizer.
        
        Args:
            wavelength: Design wavelength (meters)
            memory_limit_3k: Limit resolution to 3K per axis for memory constraints
        """
        self.wavelength = wavelength
        self.memory_limit_3k = memory_limit_3k
        self.max_resolution_per_axis = 3072 if memory_limit_3k else 8192
        
    def optimize_from_input(self, 
                           input_resolution: Tuple[int, int],
                           input_pixel_pitch: float,
                           propagation_distances: List[float], 
                           aperture_diameters: List[float],
                           focal_lengths: Optional[List[float]] = None) -> CoordinateConfig:
        """
        Optimize coordinate system starting from input image parameters.
        
        Args:
            input_resolution: Input image resolution (height, width)
            input_pixel_pitch: Input pixel size (meters)
            propagation_distances: List of free-space propagation distances (meters)
            aperture_diameters: List of aperture diameters (meters)
            focal_lengths: Optional list of lens focal lengths for validation (meters)
            
        Returns:
            Optimized coordinate configuration with warnings
        """
        warnings = []
        
        # STEP 1: Start with input image parameters
        input_field_size = (input_resolution[0] * input_pixel_pitch, 
                           input_resolution[1] * input_pixel_pitch)
        
        print(f"Input field size: {input_field_size[0]*1e3:.2f} × {input_field_size[1]*1e3:.2f} mm")
        
        # STEP 2: Calculate maximum diffraction angle from input aperture (field size)
        max_diffraction_angle = self._calc_max_diffraction_angle(input_field_size)
        print(f"Max diffraction angle: {max_diffraction_angle*1e3:.3f} mrad")
        
        # STEP 3: Calculate required padding for each propagation distance
        required_resolution, padding_info = self._calc_propagation_padding(
            input_resolution, input_pixel_pitch, propagation_distances, max_diffraction_angle)
        
        print(f"Required resolution for propagation: {required_resolution[0]} × {required_resolution[1]}")
        
        # STEP 4: Apply memory constraints and finalize resolution
        final_resolution, memory_warnings = self._apply_memory_constraints(required_resolution)
        warnings.extend(memory_warnings)
        
        # STEP 5: Validate apertures as reset points
        aperture_warnings = self._validate_aperture_reset_points(
            final_resolution, input_pixel_pitch, aperture_diameters, propagation_distances)
        warnings.extend(aperture_warnings)
        
        # STEP 6: Calculate final parameters
        final_field_size = (final_resolution[0] * input_pixel_pitch, 
                           final_resolution[1] * input_pixel_pitch)
        
        padding_factor = max(final_resolution[0] / input_resolution[0],
                           final_resolution[1] / input_resolution[1])
        
        # Calculate performance metrics
        max_distance = max(propagation_distances) if propagation_distances else 1e-3
        fresnel_number = self._calc_fresnel_number(min(final_field_size), max_distance)
        max_spatial_freq = 1.0 / (2 * input_pixel_pitch)
        
        # Use input field size as recommended sensor size (before padding)
        recommended_sensor_size = input_field_size
        
        return CoordinateConfig(
            resolution=final_resolution,
            pixel_pitch=input_pixel_pitch, 
            field_size=final_field_size,
            padding_factor=padding_factor,
            fresnel_number=fresnel_number,
            max_spatial_freq=max_spatial_freq,
            recommended_sensor_size=recommended_sensor_size,
            warnings=warnings
        )
    
    def _calc_max_diffraction_angle(self, field_size: Tuple[float, float]) -> float:
        """Calculate maximum diffraction angle based on input field size."""
        # The input field acts like an aperture - calculate its diffraction angle
        min_field_dimension = min(field_size)
        # First diffraction minimum angle: sin(θ) ≈ θ ≈ 1.22λ/D for circular aperture
        # For square aperture: θ ≈ λ/D
        max_angle = self.wavelength / min_field_dimension
        return max_angle
    
    def _calc_propagation_padding(self, input_resolution: Tuple[int, int],
                                 input_pixel_pitch: float,
                                 propagation_distances: List[float],
                                 max_diffraction_angle: float) -> Tuple[Tuple[int, int], Dict]:
        """Calculate required resolution and padding for propagation distances."""
        input_field_size = (input_resolution[0] * input_pixel_pitch,
                           input_resolution[1] * input_pixel_pitch)
        
        max_padding_needed = 1.0  # Start with no padding
        padding_info = {}
        
        for i, distance in enumerate(propagation_distances):
            if distance <= 0:
                continue
                
            # Calculate beam spreading due to diffraction over this distance
            # Spreading radius = distance * max_diffraction_angle
            spreading_radius = distance * max_diffraction_angle
            
            # Total field size needed = original + 2*spreading (both sides)
            required_field_h = input_field_size[0] + 2 * spreading_radius
            required_field_w = input_field_size[1] + 2 * spreading_radius
            
            # Calculate required resolution for this field size
            required_res_h = int(np.ceil(required_field_h / input_pixel_pitch))
            required_res_w = int(np.ceil(required_field_w / input_pixel_pitch))
            
            # Calculate padding factor needed
            padding_h = required_res_h / input_resolution[0]
            padding_w = required_res_w / input_resolution[1]
            padding_factor = max(padding_h, padding_w)
            
            max_padding_needed = max(max_padding_needed, padding_factor)
            
            padding_info[f'distance_{i+1}'] = {
                'distance': distance,
                'spreading_radius': spreading_radius,
                'padding_factor': padding_factor,
                'required_resolution': (required_res_h, required_res_w)
            }
            
            print(f"Distance {distance*1e3:.1f}mm: spreading={spreading_radius*1e6:.0f}μm, padding={padding_factor:.2f}x")
        
        # Final required resolution
        final_res_h = int(np.ceil(input_resolution[0] * max_padding_needed))
        final_res_w = int(np.ceil(input_resolution[1] * max_padding_needed))
        
        # Round up to next power of 2 for FFT efficiency
        final_res_h = 2 ** int(np.ceil(np.log2(final_res_h)))
        final_res_w = 2 ** int(np.ceil(np.log2(final_res_w)))
        
        return (final_res_h, final_res_w), padding_info
    
    def _apply_memory_constraints(self, required_resolution: Tuple[int, int]) -> Tuple[Tuple[int, int], List[str]]:
        """Apply memory constraints and return final resolution with warnings."""
        warnings = []
        final_h, final_w = required_resolution
        
        # Apply memory limits
        if final_h > self.max_resolution_per_axis:
            warnings.append(f"Required height resolution {final_h} exceeds limit {self.max_resolution_per_axis}")
            final_h = self.max_resolution_per_axis
            
        if final_w > self.max_resolution_per_axis:
            warnings.append(f"Required width resolution {final_w} exceeds limit {self.max_resolution_per_axis}")
            final_w = self.max_resolution_per_axis
            
        # Ensure minimum resolution
        final_h = max(final_h, 64)
        final_w = max(final_w, 64)
        
        return (final_h, final_w), warnings
    
    def _validate_aperture_reset_points(self, final_resolution: Tuple[int, int],
                                       pixel_pitch: float,
                                       aperture_diameters: List[float],
                                       propagation_distances: List[float]) -> List[str]:
        """Validate that apertures can act as proper reset points."""
        warnings = []
        
        # Calculate the field size that our final resolution can support
        max_field_h = final_resolution[0] * pixel_pitch
        max_field_w = final_resolution[1] * pixel_pitch
        max_field_diameter = min(max_field_h, max_field_w)
        
        print(f"Final field diameter: {max_field_diameter*1e3:.2f}mm")
        
        for i, aperture_diameter in enumerate(aperture_diameters):
            if aperture_diameter <= 0:
                continue
                
            # Check if aperture is properly sampled
            pixels_across_aperture = aperture_diameter / pixel_pitch
            min_required_pixels = 10  # Minimum 10 pixels across aperture
            
            if pixels_across_aperture < min_required_pixels:
                warnings.append(
                    f"Aperture #{i+1} diameter {aperture_diameter*1e3:.2f}mm is too small - "
                    f"only {pixels_across_aperture:.1f} pixels across (min {min_required_pixels})")
            
            # Check if aperture fits within our field
            safety_factor = 3.0  # Aperture should be <1/3 of field for safety
            if aperture_diameter > max_field_diameter / safety_factor:
                suggested_diameter = max_field_diameter / safety_factor
                warnings.append(
                    f"Aperture #{i+1} diameter {aperture_diameter*1e3:.2f}mm too large for field "
                    f"{max_field_diameter*1e3:.2f}mm. Suggest max {suggested_diameter*1e3:.2f}mm")
                
            # Check if this aperture can act as effective reset point
            if i < len(propagation_distances):
                next_distance = propagation_distances[i]
                # After passing through aperture, diffraction spreading starts fresh
                new_max_angle = self.wavelength / aperture_diameter
                print(f"Aperture #{i+1}: {aperture_diameter*1e3:.2f}mm diameter, "
                      f"new max angle: {new_max_angle*1e3:.3f}mrad")
        
        return warnings
    
    # Keep this method for backward compatibility
    def optimize_for_system(self, 
                           focal_lengths: List[float],
                           propagation_distances: List[float], 
                           aperture_diameters: List[float],
                           target_resolution: Optional[Tuple[int, int]] = None) -> CoordinateConfig:
        """Legacy method - converts to new input-driven approach."""
        # Use a default input resolution if not provided
        input_resolution = target_resolution or (128, 128)
        
        # Estimate pixel pitch from apertures and focal lengths
        if focal_lengths and aperture_diameters:
            # Conservative estimate: pixel_pitch ≈ λ*f/D for first lens
            estimated_pitch = self.wavelength * focal_lengths[0] / aperture_diameters[0] / 4
            estimated_pitch = max(estimated_pitch, 1e-6)  # At least 1μm
            estimated_pitch = min(estimated_pitch, 50e-6)  # At most 50μm
        else:
            estimated_pitch = 5e-6  # Default 5μm pixels
            
        return self.optimize_from_input(
            input_resolution=input_resolution,
            input_pixel_pitch=estimated_pitch,
            propagation_distances=propagation_distances,
            aperture_diameters=aperture_diameters,
            focal_lengths=focal_lengths
        )
    
    def _calc_fresnel_number(self, field_size: float, distance: float) -> float:
        """Calculate Fresnel number for the system."""
        if distance <= 0:
            return float('inf')
        a = field_size / 2  # Half field size
        return a**2 / (self.wavelength * distance)
    
    def _calc_recommended_sensor_size(self, field_size: Tuple[float, float],
                                     focal_lengths: Optional[List[float]]) -> Tuple[float, float]:
        """Calculate recommended physical sensor size."""
        # For imaging systems, sensor size should match image size
        return field_size
    
    def _calc_padding_factor(self, propagation_distances: List[float],
                            field_size: Tuple[float, float]) -> Tuple[float, List[str]]:
        """Calculate required padding factor for propagation - legacy method."""
        # Simple calculation for backward compatibility
        if not propagation_distances:
            return 1.5, []
            
        max_dist = max(propagation_distances)
        min_field = min(field_size)
        spreading_ratio = max_dist / min_field if min_field > 0 else 1
        padding = min(1.5 + 0.5 * spreading_ratio, 5.0)
        
        warnings = []
        if padding > 3.0:
            warnings.append(f"High padding factor ({padding:.1f}) required")
            
        return padding, warnings


def print_coordinate_config(config: CoordinateConfig, show_detailed_info: bool = True):
    """Print coordinate configuration in a readable format."""
    print("=" * 60)
    print("OPTIMIZED COORDINATE SYSTEM")
    print("=" * 60)
    print(f"Resolution: {config.resolution[0]}×{config.resolution[1]} pixels")
    print(f"Pixel pitch: {config.pixel_pitch*1e6:.2f}μm")
    print(f"Field size: {config.field_size[0]*1e3:.1f}×{config.field_size[1]*1e3:.1f}mm")
    print(f"Padding factor: {config.padding_factor:.1f}")
    print(f"Fresnel number: {config.fresnel_number:.2f}")
    print(f"Max spatial freq: {config.max_spatial_freq*1e-3:.1f} cycles/mm")
    print(f"Recommended sensor: {config.recommended_sensor_size[0]*1e3:.1f}×{config.recommended_sensor_size[1]*1e3:.1f}mm")
    
    if config.warnings:
        print("\nWARNINGS:")
        for warning in config.warnings:
            print(f"  - {warning}")
    print("=" * 60)