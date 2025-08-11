"""
ROI (Region of Interest) manager for dynamic field sizing in optical propagation.

Instead of predicting optimal coordinates upfront, this system:
1. Tracks ROI mask as information spreads during propagation
2. Expands ROI based on diffraction spreading
3. Resets ROI at apertures (natural coordinate reset points)
4. Automatically handles field sizing throughout the optical system
"""

import torch
import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass

@dataclass
class ROIState:
    """Current state of the Region of Interest."""
    center_x: float  # ROI center position (meters)
    center_y: float  # ROI center position (meters) 
    width: float     # ROI width (meters) - actual physical ROI size
    height: float    # ROI height (meters) - actual physical ROI size
    resolution: Tuple[int, int]  # True ROI resolution based on physics
    pixel_pitch: float  # Current pixel pitch (meters)
    padded_resolution: Tuple[int, int] = None  # Computational field size (power-of-2 for FFT)
    
    def __post_init__(self):
        """Set default padded_resolution if not provided."""
        if self.padded_resolution is None:
            self.padded_resolution = self.resolution
    
    @property
    def field_size(self) -> Tuple[float, float]:
        """Total field size in meters (based on padded resolution for computation)."""
        return (self.padded_resolution[0] * self.pixel_pitch, 
                self.padded_resolution[1] * self.pixel_pitch)
    
    @property 
    def roi_field_size(self) -> Tuple[float, float]:
        """Actual ROI field size in meters (based on true ROI resolution)."""
        return (self.resolution[0] * self.pixel_pitch,
                self.resolution[1] * self.pixel_pitch)
    
    @property
    def roi_bounds(self) -> Tuple[float, float, float, float]:
        """ROI bounds: (x_min, x_max, y_min, y_max) in meters."""
        x_min = self.center_x - self.width / 2
        x_max = self.center_x + self.width / 2
        y_min = self.center_y - self.height / 2
        y_max = self.center_y + self.height / 2
        return (x_min, x_max, y_min, y_max)

class ROIManager:
    """
    Manages dynamic ROI tracking throughout optical propagation.
    
    Key concept: Instead of predicting coordinates, track actual information flow:
    - ROI expands during propagation (diffraction spreading)
    - ROI contracts/resets at apertures (physical clipping)
    - Field size adapts automatically to ROI requirements
    """
    
    def __init__(self, initial_roi: ROIState, max_resolution: int = 3072):
        """
        Initialize ROI manager.
        
        Args:
            initial_roi: Starting ROI state
            max_resolution: Maximum resolution per axis for memory constraints
        """
        self.roi = initial_roi
        self.max_resolution = max_resolution
        self.roi_history = [initial_roi]
        
    def predict_propagation_roi(self, distance: float, wavelength: float, 
                               safety_factor: float = 1.2, min_expansion: bool = True) -> ROIState:
        """
        Predict ROI after free-space propagation.
        
        Args:
            distance: Propagation distance (meters)
            wavelength: Wavelength (meters)
            safety_factor: Extra margin for diffraction spreading
            min_expansion: If True, avoid unnecessary expansion for minimal diffraction
            
        Returns:
            New ROI state after propagation
        """
        # Calculate maximum diffraction angle from current ROI
        max_roi_size = max(self.roi.width, self.roi.height)
        max_diffraction_angle = wavelength / max_roi_size
        
        # Calculate spreading radius due to diffraction
        spreading_radius = distance * max_diffraction_angle * safety_factor
        
        # Expand ROI to include diffraction spreading
        new_width = self.roi.width + 2 * spreading_radius
        new_height = self.roi.height + 2 * spreading_radius
        
        # Calculate required resolution to maintain pixel pitch
        required_res_x = int(np.ceil(new_width / self.roi.pixel_pitch))
        required_res_y = int(np.ceil(new_height / self.roi.pixel_pitch))
        
        # Apply memory constraints
        required_res_x = min(required_res_x, self.max_resolution)
        required_res_y = min(required_res_y, self.max_resolution)
        
        # Separate ROI size calculation from padding strategy
        # Step 1: Calculate true ROI resolution based on physics
        true_roi_res_x = required_res_x
        true_roi_res_y = required_res_y
        
        # Step 2: Determine computational padding (reuse existing if ROI still fits)
        current_padded = self.roi.padded_resolution
        if (true_roi_res_x <= current_padded[0] and true_roi_res_y <= current_padded[1]):
            # ROI still fits in current padding - reuse it
            padded_res_x = current_padded[0] 
            padded_res_y = current_padded[1]
            print(f"  ROI {true_roi_res_x}×{true_roi_res_y} fits in existing padding {padded_res_x}×{padded_res_y}")
        else:
            # ROI exceeds current padding - need to expand with power-of-2 for FFT efficiency
            padded_res_x = 2 ** int(np.ceil(np.log2(true_roi_res_x)))
            padded_res_y = 2 ** int(np.ceil(np.log2(true_roi_res_y)))
            print(f"  ROI {true_roi_res_x}×{true_roi_res_y} exceeds padding, expanding to {padded_res_x}×{padded_res_y}")
        
        # Apply memory constraints to padding
        padded_res_x = min(padded_res_x, self.max_resolution)
        padded_res_y = min(padded_res_y, self.max_resolution)
        
        # Store both true ROI resolution and padded resolution
        required_res_x = true_roi_res_x
        required_res_y = true_roi_res_y
        
        # Ensure minimum resolution
        required_res_x = max(required_res_x, 64)
        required_res_y = max(required_res_y, 64)
        
        # Calculate actual ROI physical size based on true resolution
        actual_width = required_res_x * self.roi.pixel_pitch
        actual_height = required_res_y * self.roi.pixel_pitch
        
        return ROIState(
            center_x=self.roi.center_x,
            center_y=self.roi.center_y,
            width=actual_width,  # Physical ROI size based on diffraction
            height=actual_height,
            resolution=(required_res_x, required_res_y),  # True ROI resolution
            pixel_pitch=self.roi.pixel_pitch,
            padded_resolution=(padded_res_x, padded_res_y)  # Computational padding
        )
    
    def apply_aperture_constraint(self, aperture_diameter: float, 
                                 aperture_center: Tuple[float, float] = (0.0, 0.0)) -> ROIState:
        """
        Apply aperture constraint to ROI - this resets the coordinate system.
        
        Args:
            aperture_diameter: Aperture diameter (meters)
            aperture_center: Aperture center position (x, y) in meters
            
        Returns:
            New ROI state clipped by aperture
        """
        # Check if aperture actually constrains the field
        if self.roi.width <= aperture_diameter and self.roi.height <= aperture_diameter:
            # Aperture is larger than field - no constraint needed
            print(f"  Aperture ({aperture_diameter*1e3:.1f}mm) ≥ field ({self.roi.width*1e3:.1f}mm) - no constraint")
            return ROIState(
                center_x=self.roi.center_x,
                center_y=self.roi.center_y,
                width=self.roi.width,
                height=self.roi.height,
                resolution=self.roi.resolution,
                pixel_pitch=self.roi.pixel_pitch,
                padded_resolution=self.roi.padded_resolution
            )
        
        # Aperture constrains the field
        print(f"  Aperture ({aperture_diameter*1e3:.1f}mm) < field ({self.roi.width*1e3:.1f}mm) - constraining")
        
        # New ROI is constrained by aperture size
        new_width = min(self.roi.width, aperture_diameter)
        new_height = min(self.roi.height, aperture_diameter)
        
        # Center ROI on aperture
        new_center_x = aperture_center[0]
        new_center_y = aperture_center[1]
        
        # Keep current pixel pitch for consistency
        new_pixel_pitch = self.roi.pixel_pitch
        
        # Calculate resolution for new ROI
        required_res_x = int(np.ceil(new_width / new_pixel_pitch))
        required_res_y = int(np.ceil(new_height / new_pixel_pitch))
        
        # Apply limits
        required_res_x = max(64, min(required_res_x, self.max_resolution))
        required_res_y = max(64, min(required_res_y, self.max_resolution))
        
        # Recalculate actual field size
        actual_width = required_res_x * new_pixel_pitch
        actual_height = required_res_y * new_pixel_pitch
        
        # For aperture constraint, keep current padding since aperture should not change computational field size
        return ROIState(
            center_x=new_center_x,
            center_y=new_center_y,
            width=actual_width,
            height=actual_height,
            resolution=(required_res_x, required_res_y),
            pixel_pitch=new_pixel_pitch,
            padded_resolution=self.roi.padded_resolution  # Keep existing padding
        )
    
    def update_roi(self, new_roi: ROIState, description: str = ""):
        """Update current ROI and save to history."""
        self.roi = new_roi
        self.roi_history.append(new_roi)
        
        if description:
            print(f"ROI Update - {description}:")
            print(f"  Resolution: {new_roi.resolution[0]}×{new_roi.resolution[1]}")
            print(f"  Field size: {new_roi.field_size[0]*1e3:.2f}×{new_roi.field_size[1]*1e3:.2f}mm")
            print(f"  ROI size: {new_roi.width*1e3:.2f}×{new_roi.height*1e3:.2f}mm")
            print(f"  Pixel pitch: {new_roi.pixel_pitch*1e6:.1f}μm")
    
    def pad_field_to_roi(self, field: torch.Tensor, current_roi: ROIState, 
                        target_roi: ROIState) -> torch.Tensor:
        """
        Pad/crop field from current ROI to match target ROI.
        
        Args:
            field: Current field tensor
            current_roi: Current ROI state  
            target_roi: Target ROI state
            
        Returns:
            Field resized to target ROI
        """
        if current_roi.resolution == target_roi.resolution:
            return field
        
        # Calculate padding/cropping needed
        current_h, current_w = current_roi.resolution
        target_h, target_w = target_roi.resolution
        
        if target_h >= current_h and target_w >= current_w:
            # Need to pad
            pad_h = (target_h - current_h) // 2
            pad_w = (target_w - current_w) // 2
            
            import torch.nn.functional as F
            field = F.pad(field, (pad_w, pad_w, pad_h, pad_h), mode='constant', value=0)
            
        elif target_h <= current_h and target_w <= current_w:
            # Need to crop
            crop_h = (current_h - target_h) // 2
            crop_w = (current_w - target_w) // 2
            
            field = field[crop_h:crop_h+target_h, crop_w:crop_w+target_w]
            
        else:
            # Mixed pad/crop - handle as needed
            # For now, use interpolation as fallback
            import torch.nn.functional as F
            field = F.interpolate(
                field.unsqueeze(0).unsqueeze(0), 
                size=(target_h, target_w), 
                mode='bilinear', 
                align_corners=False
            ).squeeze(0).squeeze(0)
        
        return field
    
    def get_current_config(self) -> dict:
        """Get current ROI as system configuration parameters."""
        return {
            'sensor_resolution': self.roi.resolution,
            'pixel_pitch': self.roi.pixel_pitch,
            'field_size': self.roi.field_size,
            'roi_center': (self.roi.center_x, self.roi.center_y),
            'roi_size': (self.roi.width, self.roi.height)
        }
    
    def print_roi_history(self):
        """Print the evolution of ROI throughout the optical system."""
        print("\nROI Evolution History:")
        print("-" * 80)
        for i, roi in enumerate(self.roi_history):
            print(f"Step {i}: {roi.resolution[0]}×{roi.resolution[1]} @ {roi.pixel_pitch*1e6:.1f}μm")
            print(f"         Field: {roi.field_size[0]*1e3:.1f}×{roi.field_size[1]*1e3:.1f}mm")
            print(f"         ROI: {roi.width*1e3:.1f}×{roi.height*1e3:.1f}mm")
        print("-" * 80)