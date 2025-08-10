# Fixed system/imaging_system.py - NO relative imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional, Any

class ImagingSystem(nn.Module):
    def __init__(self, config, elements, element_positions):
        super().__init__()
        self.config = config
        self.elements = nn.ModuleList(elements)
        self.element_positions = element_positions
        self.cache_intermediates = False
        self._intermediate_fields = []
        
        # Import here to avoid issues
        from core.propagation import OpticalPropagation
        from system.sensor_model import SensorModel
        
        self.propagator = OpticalPropagation()
        self.sensor = SensorModel(config)
    
    def forward(self, intensity_image, phase_image=None, wavelength=None):
        self._intermediate_fields = []
        
        if wavelength is None:
            wavelength = self.config.wavelength
        
        if phase_image is None:
            phase_image = torch.zeros_like(intensity_image)
        
        amplitude = torch.sqrt(intensity_image + 1e-10)
        field = amplitude * torch.exp(1j * phase_image)
        
        if self.cache_intermediates:
            self._intermediate_fields.append(field.clone())
        
        current_position = 0.0
        for element, position in zip(self.elements, self.element_positions):
            if position > current_position:
                distance = position - current_position
                field = self.propagator.propagate(
                    field, distance, wavelength, self.config.pixel_pitch,
                    method=self.config.propagation_method.value
                )
                current_position = position
            
            field = element(field)
            
            if self.cache_intermediates:
                self._intermediate_fields.append(field.clone())
        
        if self.config.propagation_distance > current_position:
            distance = self.config.propagation_distance - current_position
            field = self.propagator.propagate(
                field, distance, wavelength, self.config.pixel_pitch,
                method=self.config.propagation_method.value
            )
        
        sensor_output = self.sensor(field)
        
        return {
            'intensity_sensor': sensor_output['intensity'],
            'field_sensor': field,
            'snr': sensor_output['snr'],
            'intermediate_fields': self._intermediate_fields
        }
    
    def set_cache_intermediates(self, cache):
        self.cache_intermediates = cache