import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .base import OpticalElement


class CodedAperture(OpticalElement):
    
    def __init__(self, resolution, pixel_pitch, wavelength, aperture_type='binary',
                 learnable=True, name=None):
        self.aperture_type = aperture_type
        self.learnable = learnable
        super().__init__(resolution, pixel_pitch, wavelength, name)
    
    def _init_parameters(self):
        H, W = self.resolution
        
        if self.aperture_type == 'mura':
            pattern = self._create_mura_pattern(H, W)
        else:
            pattern = torch.randn(1, H, W) * 0.1
        
        if self.learnable:
            self.pattern_logits = nn.Parameter(pattern)
        else:
            self.register_buffer('pattern_logits', pattern)
    
    def _create_mura_pattern(self, H, W):
        size = min(H, W)
        # Find nearest prime
        def is_prime(n):
            if n < 2:
                return False
            for i in range(2, int(np.sqrt(n)) + 1):
                if n % i == 0:
                    return False
            return True
        
        while not is_prime(size):
            size -= 1
        
        pattern = torch.zeros(size, size)
        for i in range(size):
            for j in range(size):
                if i == 0:
                    pattern[i, j] = 1
                elif j == 0:
                    pattern[i, j] = 0
                else:
                    if (i * j) % size == 1:
                        pattern[i, j] = 1
        
        pattern = pattern.unsqueeze(0).unsqueeze(0)
        pattern = F.interpolate(pattern, size=(H, W), mode='nearest')
        return pattern.squeeze(0)
    
    def forward(self, field):
        if self.aperture_type == 'binary':
            if self.training and self.learnable:
                transmittance = torch.sigmoid(10 * self.pattern_logits)
            else:
                transmittance = (self.pattern_logits > 0).float()
        else:
            transmittance = torch.sigmoid(self.pattern_logits)
        
        return field * transmittance
    
    def visualize(self):
        return {'amplitude': torch.sigmoid(self.pattern_logits).squeeze(0)}