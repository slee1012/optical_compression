"""
ROI-based imaging system that dynamically tracks field size during propagation.

This replaces the complex upfront coordinate optimization with a simpler,
more intuitive approach that follows actual information flow.
"""

import torch
import torch.nn as nn
from typing import List, Dict, Optional, Tuple
from physics.roi_manager import ROIManager, ROIState
from physics.propagation import OpticalPropagation

class ROIBasedImagingSystem(nn.Module):
    """
    Imaging system with dynamic ROI tracking.
    
    Key differences from traditional approach:
    1. No upfront coordinate prediction
    2. ROI expands during propagation (diffraction spreading)
    3. ROI resets at apertures (natural coordinate reset)
    4. Field size adapts automatically to information flow
    """
    
    def __init__(self, initial_resolution: Tuple[int, int], initial_pixel_pitch: float,
                 wavelength: float, element_specs: List[Dict], element_positions: List[float],
                 element_spacings: List[float], element_apertures: List[float]):
        super().__init__()
        
        self.wavelength = wavelength
        self.element_specs = element_specs  # Store element specifications instead of pre-built elements
        self.element_positions = element_positions
        self.element_spacings = element_spacings
        self.element_apertures = element_apertures
        
        # Initialize ROI manager with input field parameters
        initial_roi = ROIState(
            center_x=0.0,
            center_y=0.0,
            width=initial_resolution[0] * initial_pixel_pitch,
            height=initial_resolution[1] * initial_pixel_pitch,
            resolution=initial_resolution,
            pixel_pitch=initial_pixel_pitch
        )
        
        self.roi_manager = ROIManager(initial_roi)
        
        # Sensor model
        from simulation.sensor_model import SensorModel
        self.sensor = None  # Will be created dynamically based on final ROI
        
        # Caching
        self.cache_intermediates = False
        self._intermediate_fields = []
        self._intermediate_rois = []
        self._intermediate_labels = []
    
    @property
    def elements(self):
        """Compatibility property: return element_specs for external access."""
        return self.element_specs
    
    def forward(self, input_field: torch.Tensor) -> Dict:
        """
        Propagate field through system with dynamic ROI tracking.
        
        Args:
            input_field: Input complex field
            
        Returns:
            Dictionary with sensor output and intermediate data
        """
        self._intermediate_fields = []
        self._intermediate_rois = []  
        self._intermediate_labels = []
        
        # Start with input field and ROI
        current_field = input_field
        current_position = 0.0
        
        # Cache input
        if self.cache_intermediates:
            self._intermediate_fields.append(current_field.clone())
            self._intermediate_rois.append(self.roi_manager.roi)
            self._intermediate_labels.append("Input Field")
        
        print("\\n=== ROI-BASED PROPAGATION ===")
        
        for i, (element_spec, position) in enumerate(zip(self.element_specs, self.element_positions)):
            
            # Step 1: Propagate to element position if needed
            if position > current_position:
                distance = position - current_position
                
                # Predict ROI after propagation
                propagation_roi = self.roi_manager.predict_propagation_roi(
                    distance, self.wavelength, safety_factor=1.2
                )
                
                print(f"\\nElement {i+1}: Propagating {distance*1e3:.1f}mm to {element_spec['type']}")
                self.roi_manager.update_roi(propagation_roi, f"Before propagation to element {i+1}")
                
                # Resize field to match new ROI
                current_field = self.roi_manager.pad_field_to_roi(
                    current_field, self.roi_manager.roi_history[-2], propagation_roi
                )
                
                # Propagate with no additional padding (ROI already sized correctly)
                current_field = OpticalPropagation.propagate(
                    current_field, distance, self.wavelength, 
                    propagation_roi.pixel_pitch, padding_factor=1.0
                )
                
                current_position = position
                
                # Cache field after propagation
                if self.cache_intermediates:
                    self._intermediate_fields.append(current_field.clone())
                    self._intermediate_rois.append(propagation_roi)
                    self._intermediate_labels.append(f"Before {element_spec['type']} #{i+1}")
            
            # Step 2: Create and apply optical element dynamically
            print(f"  Creating {element_spec['type']} for ROI: {self.roi_manager.roi.resolution} (padded: {self.roi_manager.roi.padded_resolution})")
            print(f"  Current field shape: {current_field.shape}")
            
            # Ensure field size matches computational padding (not just ROI resolution)
            target_shape = self.roi_manager.roi.padded_resolution
            if current_field.shape[-2:] != target_shape:
                print(f"  Resizing field {current_field.shape[-2:]} → {target_shape}")
                # Handle complex tensor interpolation by processing real and imaginary parts
                import torch.nn.functional as F
                
                real_part = F.interpolate(
                    current_field.real.unsqueeze(0).unsqueeze(0), 
                    size=target_shape, 
                    mode='bilinear', 
                    align_corners=False
                ).squeeze(0).squeeze(0)
                
                imag_part = F.interpolate(
                    current_field.imag.unsqueeze(0).unsqueeze(0), 
                    size=target_shape, 
                    mode='bilinear', 
                    align_corners=False
                ).squeeze(0).squeeze(0)
                
                current_field = torch.complex(real_part, imag_part)
            
            element = self._create_element_at_runtime(element_spec, self.roi_manager.roi)
            current_field = element(current_field)
            
            # Step 3: Apply aperture constraint (coordinate reset)
            if self.element_apertures[i] < float('inf'):
                aperture_roi = self.roi_manager.apply_aperture_constraint(
                    self.element_apertures[i], aperture_center=(0.0, 0.0)
                )
                
                self.roi_manager.update_roi(aperture_roi, f"After aperture {i+1}")
                
                # Resize field to aperture-constrained ROI
                current_field = self.roi_manager.pad_field_to_roi(
                    current_field, self.roi_manager.roi_history[-2], aperture_roi
                )
                
                # Apply circular aperture mask
                current_field = self._apply_circular_aperture(
                    current_field, self.element_apertures[i], aperture_roi.pixel_pitch
                )
            
            # Cache field after element
            if self.cache_intermediates:
                self._intermediate_fields.append(current_field.clone())
                self._intermediate_rois.append(self.roi_manager.roi)
                self._intermediate_labels.append(f"After {element_spec['type']} #{i+1}")
            
            # Step 4: Propagate through element spacing if specified
            if self.element_spacings[i] > 0:
                spacing_distance = self.element_spacings[i]
                
                # Predict ROI after spacing propagation
                spacing_roi = self.roi_manager.predict_propagation_roi(
                    spacing_distance, self.wavelength, safety_factor=1.2
                )
                
                self.roi_manager.update_roi(spacing_roi, f"After {spacing_distance*1e3:.1f}mm spacing")
                
                # Resize field for spacing propagation
                current_field = self.roi_manager.pad_field_to_roi(
                    current_field, self.roi_manager.roi_history[-2], spacing_roi
                )
                
                # Propagate
                current_field = OpticalPropagation.propagate(
                    current_field, spacing_distance, self.wavelength,
                    spacing_roi.pixel_pitch, padding_factor=1.0
                )
                
                current_position += spacing_distance
                
                # Cache field after spacing
                if self.cache_intermediates:
                    self._intermediate_fields.append(current_field.clone())
                    self._intermediate_rois.append(spacing_roi)
                    self._intermediate_labels.append(f"After {spacing_distance*1e3:.1f}mm spacing from {element_spec['type']} #{i+1}")
        
        # Final sensor simulation
        final_roi = self.roi_manager.roi
        
        # Create sensor model dynamically based on final padded field size
        sensor_config = type('Config', (), {
            'sensor_resolution': final_roi.padded_resolution,  # Use padded size for computation
            'pixel_pitch': final_roi.pixel_pitch,
            'wavelength': self.wavelength
        })()
        
        from simulation.sensor_model import SensorModel
        sensor = SensorModel(sensor_config)
        sensor_output = sensor(current_field)
        
        # IMPORTANT: Crop sensor output back to original input resolution
        original_roi = self.roi_manager.roi_history[0]  # Get initial ROI
        cropped_intensity = self._crop_to_original_resolution(
            sensor_output['intensity'], final_roi, original_roi
        )
        cropped_field = self._crop_to_original_resolution(
            current_field, final_roi, original_roi
        )
        
        print(f"\\nFinal ROI: {final_roi.resolution[0]}×{final_roi.resolution[1]} (padded: {final_roi.padded_resolution[0]}×{final_roi.padded_resolution[1]}) @ {final_roi.pixel_pitch*1e6:.1f}μm")
        print(f"Cropped back to original: {original_roi.resolution[0]}×{original_roi.resolution[1]} @ {original_roi.pixel_pitch*1e6:.1f}μm")
        print("=== ROI-BASED PROPAGATION COMPLETE ===\\n")
        
        return {
            'intensity_sensor': cropped_intensity,  # Cropped to original size
            'field_sensor': cropped_field,          # Cropped field
            'snr': sensor_output['snr'],
            'intermediate_fields': self._intermediate_fields,
            'intermediate_rois': self._intermediate_rois,
            'intermediate_labels': self._intermediate_labels,
            'final_roi': final_roi,
            'original_roi': original_roi,
            'roi_history': self.roi_manager.roi_history.copy(),
            # Also provide uncropped versions for analysis
            'intensity_sensor_full': sensor_output['intensity'],
            'field_sensor_full': current_field
        }
    
    def _apply_circular_aperture(self, field: torch.Tensor, aperture_diameter: float, 
                                pixel_pitch: float) -> torch.Tensor:
        """Apply circular aperture mask to field."""
        H, W = field.shape[-2:]
        
        # Create circular aperture mask
        y, x = torch.meshgrid(
            torch.linspace(-1, 1, H, device=field.device),
            torch.linspace(-1, 1, W, device=field.device),
            indexing='ij'
        )
        
        # Convert aperture diameter to normalized coordinates
        field_size = min(H * pixel_pitch, W * pixel_pitch)
        normalized_radius = aperture_diameter / field_size
        
        # Create mask
        radius_map = torch.sqrt(x**2 + y**2)
        aperture_mask = (radius_map <= normalized_radius).float()
        
        return field * aperture_mask
    
    def set_cache_intermediates(self, cache: bool):
        """Enable/disable intermediate field caching."""
        self.cache_intermediates = cache
    
    def get_roi_summary(self) -> Dict:
        """Get summary of ROI evolution."""
        return {
            'initial_roi': self.roi_manager.roi_history[0],
            'final_roi': self.roi_manager.roi,
            'num_steps': len(self.roi_manager.roi_history),
            'roi_history': self.roi_manager.roi_history.copy()
        }
    
    def print_roi_evolution(self):
        """Print detailed ROI evolution."""
        self.roi_manager.print_roi_history()
    
    def _create_element_at_runtime(self, element_spec: Dict, roi_state: ROIState):
        """Create optical element at runtime with current ROI parameters."""
        element_type = element_spec['type']
        
        if element_type == 'ThinLens':
            from elements.lens import create_thin_lens
            
            # The key insight: aperture_diameter is in physical units, 
            # but we create the lens with the current ROI's resolution and pixel pitch
            element = create_thin_lens(
                focal_length=element_spec['focal_length'],
                aperture_diameter=element_spec['aperture_diameter'], 
                resolution=roi_state.padded_resolution,  # Use computational padding size
                pixel_pitch=roi_state.pixel_pitch,
                wavelength=self.wavelength
            )
            return element
            
        elif element_type == 'PhaseMask':
            from elements.phase_mask import PhaseMask
            element = PhaseMask(
                resolution=roi_state.padded_resolution,  # Use computational padding size
                pixel_pitch=roi_state.pixel_pitch,
                wavelength=self.wavelength,
                learnable=element_spec.get('learnable', True)
            )
            return element
            
        else:
            raise ValueError(f"Unknown element type: {element_type}")
    
    def _crop_to_original_resolution(self, tensor: torch.Tensor, current_roi: ROIState, 
                                   original_roi: ROIState) -> torch.Tensor:
        """Crop tensor from current padded field back to original input resolution."""
        # Use padded resolution since that's the actual tensor size
        current_h, current_w = current_roi.padded_resolution
        original_h, original_w = original_roi.resolution
        
        if current_h == original_h and current_w == original_w:
            # No cropping needed
            return tensor
        
        # Calculate center crop coordinates
        crop_h = (current_h - original_h) // 2
        crop_w = (current_w - original_w) // 2
        
        # Ensure we don't crop more than available
        if crop_h < 0 or crop_w < 0:
            print(f"Warning: Cannot crop {current_roi.padded_resolution} to {original_roi.resolution}")
            return tensor
        
        # Crop to original size (centered)
        cropped = tensor[crop_h:crop_h+original_h, crop_w:crop_w+original_w]
        
        print(f"  Cropped field from {current_roi.padded_resolution} to {original_roi.resolution}")
        return cropped