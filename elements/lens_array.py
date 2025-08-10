import torch
import torch.nn as nn
import numpy as np
from .base import OpticalElement


class LensArray(OpticalElement):
    
    def __init__(self, resolution, pixel_pitch, wavelength, n_lenses=(8, 8),
                 focal_length=None, lens_type='spherical', name=None):
        self.n_lenses = n_lenses
        self.focal_length_init = focal_length
        self.lens_type = lens_type
        super().__init__(resolution, pixel_pitch, wavelength, name)
    
    def _init_parameters(self):
        H, W = self.resolution
        n_v, n_h = self.n_lenses
        self.lens_size = (H // n_v, W // n_h)
        
        if self.focal_length_init is None:
            f_init = self.lens_size[0] * self.pixel_pitch * 2
        else:
            f_init = self.focal_length_init
        
        self.register_buffer('focal_length', torch.tensor(f_init))
        self._update_lens_pattern()
    
    def _update_lens_pattern(self):
        H, W = self.resolution
        n_v, n_h = self.n_lenses
        lens_h, lens_w = self.lens_size
        
        x = torch.linspace(-1, 1, lens_w)
        y = torch.linspace(-1, 1, lens_h)
        X, Y = torch.meshgrid(x, y, indexing='xy')
        r_squared = X**2 + Y**2
        
        single_phase = -self.k * r_squared * (lens_w * self.pixel_pitch)**2 / (4 * self.focal_length)
        
        phase_array = torch.zeros(1, H, W)
        for i in range(n_v):
            for j in range(n_h):
                v_start = i * lens_h
                v_end = min((i + 1) * lens_h, H)
                h_start = j * lens_w
                h_end = min((j + 1) * lens_w, W)
                
                v_size = v_end - v_start
                h_size = h_end - h_start
                
                phase_array[0, v_start:v_end, h_start:h_end] = single_phase[:v_size, :h_size]
        
        self.register_buffer('phase_pattern', phase_array)
    
    def forward(self, field):
        field = self.ensure_complex(field)
        phase_modulation = torch.exp(1j * self.phase_pattern)
        return field * phase_modulation
    
    def visualize(self):
        return {'phase': self.phase_pattern.squeeze(0)}