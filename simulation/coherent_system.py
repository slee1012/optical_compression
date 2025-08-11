# Fixed system/imaging_system.py - NO relative imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional, Any

class ImagingSystem(nn.Module):
    def __init__(self, config, elements, element_positions,
                 element_spacings=None, element_apertures=None, refractive_indices=None):
        super().__init__()
        self.config = config
        self.elements = nn.ModuleList(elements)
        self.element_positions = element_positions
        
        # Enhanced positioning information
        self.element_spacings = element_spacings or [0.0] * len(elements)
        self.element_apertures = element_apertures or [float('inf')] * len(elements)  
        self.refractive_indices = refractive_indices or [1.0] * len(elements)
        
        self.cache_intermediates = False
        self._intermediate_fields = []
        self._intermediate_positions = []  # Track z-positions of intermediate fields
        self._intermediate_labels = []     # Track descriptive labels for each field
        
        # Import here to avoid issues
        from simulation.sensor_model import SensorModel
        
        self.sensor = SensorModel(config)
    
    def forward(self, input_data, phase_image=None, wavelength=None):
        from physics.propagation import OpticalPropagation
        
        self._intermediate_fields = []
        self._intermediate_positions = []
        self._intermediate_labels = []
        
        if wavelength is None:
            wavelength = self.config.wavelength
        
        # Handle both complex fields and intensity+phase inputs
        if torch.is_complex(input_data):
            # Input is already a complex field
            field = input_data
        else:
            # Input is intensity image, construct complex field
            intensity_image = input_data
            if phase_image is None:
                phase_image = torch.zeros_like(intensity_image)
            
            amplitude = torch.sqrt(intensity_image + 1e-10)
            field = amplitude * torch.exp(1j * phase_image)
        
        if self.cache_intermediates:
            self._intermediate_fields.append(field.clone())
            self._intermediate_positions.append(0.0)
            self._intermediate_labels.append("Input Field")
        
        current_position = 0.0
        for i, (element, position) in enumerate(zip(self.elements, self.element_positions)):
            # Propagate to element position if needed
            if position > current_position:
                distance = position - current_position
                
                # Use refractive index of previous medium (or air if first)
                medium_index = self.refractive_indices[i-1] if i > 0 else 1.0
                effective_wavelength = wavelength / medium_index
                
                field = OpticalPropagation.propagate(
                    field, distance, effective_wavelength, self.config.pixel_pitch,
                    method=self.config.propagation_method.value
                )
                current_position = position
                
                # Cache field after propagation to element
                if self.cache_intermediates:
                    self._intermediate_fields.append(field.clone())
                    self._intermediate_positions.append(current_position)
                    self._intermediate_labels.append(f"Before {type(element).__name__} #{i+1}")
            
            # Apply optical element
            field = element(field)
            
            # Apply aperture limiting if specified
            if self.element_apertures[i] < float('inf'):
                field = self._apply_circular_aperture(field, self.element_apertures[i])
            
            if self.cache_intermediates:
                self._intermediate_fields.append(field.clone())
                self._intermediate_positions.append(current_position)
                self._intermediate_labels.append(f"After {type(element).__name__} #{i+1}")
            
            # Propagate through element spacing if specified
            if self.element_spacings[i] > 0:
                spacing_distance = self.element_spacings[i]
                medium_index = self.refractive_indices[i]
                effective_wavelength = wavelength / medium_index
                
                field = OpticalPropagation.propagate(
                    field, spacing_distance, effective_wavelength, self.config.pixel_pitch,
                    method=self.config.propagation_method.value
                )
                current_position += spacing_distance
                
                # Cache field after element spacing propagation
                if self.cache_intermediates:
                    self._intermediate_fields.append(field.clone())
                    self._intermediate_positions.append(current_position)
                    self._intermediate_labels.append(f"After {spacing_distance*1e3:.1f}mm spacing from {type(element).__name__} #{i+1}")
        
        # Final propagation to sensor
        if self.config.propagation_distance > current_position:
            distance = self.config.propagation_distance - current_position
            # Use last medium's refractive index (or air)
            medium_index = self.refractive_indices[-1] if self.refractive_indices else 1.0
            effective_wavelength = wavelength / medium_index
            
            field = OpticalPropagation.propagate(
                field, distance, effective_wavelength, self.config.pixel_pitch,
                method=self.config.propagation_method.value
            )
            
            # Cache final field before sensor
            if self.cache_intermediates:
                self._intermediate_fields.append(field.clone())
                self._intermediate_positions.append(self.config.propagation_distance)
                self._intermediate_labels.append(f"At Sensor (after {distance*1e3:.1f}mm final propagation)")
        
        sensor_output = self.sensor(field)
        
        return {
            'intensity_sensor': sensor_output['intensity'],
            'field_sensor': field,
            'snr': sensor_output['snr'],
            'intermediate_fields': self._intermediate_fields,
            'intermediate_positions': self._intermediate_positions,
            'intermediate_labels': self._intermediate_labels
        }
    
    def set_cache_intermediates(self, cache):
        self.cache_intermediates = cache
    
    def get_intermediate_field(self, position=None, label_contains=None, index=None):
        """
        Get intermediate field by position, label content, or index.
        
        Args:
            position: Z-position (meters) to find closest field
            label_contains: String that should be contained in the label
            index: Direct index into intermediate fields list
        
        Returns:
            dict with 'field', 'position', 'label' or None if not found
        """
        if not self._intermediate_fields:
            return None
            
        if index is not None:
            if 0 <= index < len(self._intermediate_fields):
                return {
                    'field': self._intermediate_fields[index],
                    'position': self._intermediate_positions[index],
                    'label': self._intermediate_labels[index]
                }
            return None
            
        if position is not None:
            # Find closest position
            min_dist = float('inf')
            best_idx = None
            for i, pos in enumerate(self._intermediate_positions):
                dist = abs(pos - position)
                if dist < min_dist:
                    min_dist = dist
                    best_idx = i
            
            if best_idx is not None:
                return {
                    'field': self._intermediate_fields[best_idx],
                    'position': self._intermediate_positions[best_idx],
                    'label': self._intermediate_labels[best_idx]
                }
            return None
            
        if label_contains is not None:
            # Find first label containing the string
            for i, label in enumerate(self._intermediate_labels):
                if label_contains.lower() in label.lower():
                    return {
                        'field': self._intermediate_fields[i],
                        'position': self._intermediate_positions[i],
                        'label': self._intermediate_labels[i]
                    }
            return None
            
        return None
    
    def list_intermediate_fields(self):
        """List all available intermediate fields with their positions and labels."""
        if not self._intermediate_fields:
            print("No intermediate fields cached. Set system.set_cache_intermediates(True) before running.")
            return
            
        print("Available intermediate fields:")
        print("-" * 70)
        for i, (pos, label) in enumerate(zip(self._intermediate_positions, self._intermediate_labels)):
            field = self._intermediate_fields[i]
            energy = (torch.abs(field)**2).sum().item()
            peak = (torch.abs(field)**2).max().item()
            print(f"{i:2d}: {pos*1e3:6.1f}mm - {label}")
            print(f"     Energy: {energy:8.1f}, Peak: {peak:.3f}")
        print("-" * 70)
    
    def _apply_circular_aperture(self, field: torch.Tensor, aperture_diameter: float) -> torch.Tensor:
        """Apply circular aperture limiting to field."""
        H, W = field.shape[-2:]
        
        # Create circular aperture mask
        y, x = torch.meshgrid(
            torch.linspace(-1, 1, H, device=field.device),
            torch.linspace(-1, 1, W, device=field.device),
            indexing='ij'
        )
        
        # Convert aperture diameter to normalized coordinates
        # Assume the full sensor width corresponds to the physical sensor size
        normalized_radius = aperture_diameter / (self.config.pixel_pitch * min(H, W))
        
        # Create mask
        radius = torch.sqrt(x**2 + y**2)
        mask = (radius <= normalized_radius).float()
        
        return field * mask
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get information about the optical system layout."""
        return {
            'n_elements': len(self.elements),
            'element_positions': self.element_positions,
            'element_spacings': self.element_spacings,
            'element_apertures': self.element_apertures,
            'refractive_indices': self.refractive_indices,
            'total_system_length': max(self.element_positions) if self.element_positions else 0.0,
            'config': self.config
        }