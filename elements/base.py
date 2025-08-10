import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple
import numpy as np


class OpticalElement(nn.Module, ABC):
    
    def __init__(self, resolution, pixel_pitch, wavelength, name=None):
        super().__init__()
        self.resolution = resolution
        self.pixel_pitch = pixel_pitch
        self.wavelength = wavelength
        self.name = name or self.__class__.__name__
        self.physical_size = (resolution[0] * pixel_pitch, resolution[1] * pixel_pitch)
        self.k = 2 * np.pi / wavelength
        self._init_parameters()
    
    @abstractmethod
    def _init_parameters(self):
        pass
    
    @abstractmethod
    def forward(self, field):
        pass
    
    def ensure_complex(self, field):
        if not torch.is_complex(field):
            return field.to(torch.complex64)
        return field
    
    def get_params_dict(self):
        return {
            'name': self.name,
            'resolution': self.resolution,
            'pixel_pitch': self.pixel_pitch,
            'wavelength': self.wavelength,
            'physical_size': self.physical_size
        }
    
    def visualize(self):
        return {}