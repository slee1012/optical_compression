import torch
import torch.nn as nn
import numpy as np
from .base import OpticalElement


class PhaseMask(OpticalElement):
    
    def __init__(self, resolution, pixel_pitch, wavelength, learnable=True, 
                 init_type='random', max_phase=2*np.pi, name=None):
        self.learnable = learnable
        self.init_type = init_type
        self.max_phase = max_phase
        super().__init__(resolution, pixel_pitch, wavelength, name)
    
    def _init_parameters(self):
        H, W = self.resolution
        
        if self.init_type == 'random':
            phase_init = torch.randn(1, H, W) * 0.1
        elif self.init_type == 'zeros':
            phase_init = torch.zeros(1, H, W)
        elif self.init_type == 'quadratic':
            x = torch.linspace(-1, 1, W)
            y = torch.linspace(-1, 1, H)
            X, Y = torch.meshgrid(x, y, indexing='xy')
            r_squared = X**2 + Y**2
            phase_init = -self.k * r_squared * (W * self.pixel_pitch)**2 / 100
            phase_init = phase_init.unsqueeze(0)
        else:
            phase_init = torch.zeros(1, H, W)
        
        if self.learnable:
            self.phase = nn.Parameter(phase_init)
        else:
            self.register_buffer('phase', phase_init)
    
    def forward(self, field):
        field = self.ensure_complex(field)
        phase = torch.clamp(self.phase, -self.max_phase, self.max_phase)
        phase_modulation = torch.exp(1j * phase)
        return field * phase_modulation
    
    def visualize(self):
        return {'phase': self.phase.squeeze(0)}