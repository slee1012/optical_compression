import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional
from enum import Enum


class PropagationType(Enum):
    NEAR_FIELD = "near_field"
    FRESNEL = "fresnel"
    FRAUNHOFER = "fraunhofer"


class OpticalPropagation:
    
    @staticmethod
    def angular_spectrum(field, distance, wavelength, pixel_pitch, padding_factor=1.5):
        squeeze_batch = False
        if field.dim() == 2:
            field = field.unsqueeze(0)
            squeeze_batch = True
        
        B, H, W = field.shape
        device = field.device if hasattr(field, 'device') else torch.device('cpu')
        
        if padding_factor > 1:
            pad_h = int(H * (padding_factor - 1) / 2)
            pad_w = int(W * (padding_factor - 1) / 2)
            field = F.pad(field, (pad_w, pad_w, pad_h, pad_h), mode='constant')
            H_pad, W_pad = field.shape[-2:]
        else:
            H_pad, W_pad = H, W
            pad_h = pad_w = 0
        
        fx = torch.fft.fftfreq(W_pad, pixel_pitch, device=device)
        fy = torch.fft.fftfreq(H_pad, pixel_pitch, device=device)
        FX, FY = torch.meshgrid(fx, fy, indexing='xy')
        
        k = 2 * np.pi / wavelength
        freq_sq = (2 * np.pi * FX)**2 + (2 * np.pi * FY)**2
        
        mask = freq_sq <= k**2
        kz = torch.sqrt(torch.clamp(k**2 - freq_sq, min=0))
        H_transfer = torch.where(mask, torch.exp(1j * kz * distance), 
                                torch.zeros_like(kz, dtype=torch.complex64))
        
        field_fft = torch.fft.fft2(field)
        field_prop_fft = field_fft * H_transfer.unsqueeze(0)
        field_prop = torch.fft.ifft2(field_prop_fft)
        
        if padding_factor > 1:
            field_prop = field_prop[..., pad_h:-pad_h if pad_h > 0 else None, 
                                   pad_w:-pad_w if pad_w > 0 else None]
        
        return field_prop.squeeze(0) if squeeze_batch else field_prop
    
    @staticmethod
    def fresnel_transfer(field, distance, wavelength, pixel_pitch):
        squeeze_batch = False
        if field.dim() == 2:
            field = field.unsqueeze(0)
            squeeze_batch = True
        
        B, H, W = field.shape
        device = field.device if hasattr(field, 'device') else torch.device('cpu')
        k = 2 * np.pi / wavelength
        
        fx = torch.fft.fftfreq(W, pixel_pitch, device=device)
        fy = torch.fft.fftfreq(H, pixel_pitch, device=device)
        FX, FY = torch.meshgrid(fx, fy, indexing='xy')
        
        H_transfer = torch.exp(-1j * np.pi * wavelength * distance * (FX**2 + FY**2))
        
        field_fft = torch.fft.fft2(field)
        field_prop_fft = field_fft * H_transfer.unsqueeze(0)
        field_prop = torch.fft.ifft2(field_prop_fft)
        
        phase = torch.tensor(k * distance, dtype=field_prop.dtype, device=device)
        field_prop = field_prop * torch.exp(1j * phase)
        
        return field_prop.squeeze(0) if squeeze_batch else field_prop
    
    @staticmethod
    def propagate(field, distance, wavelength, pixel_pitch, method=None):
        if method == 'angular' or method == 'angular_spectrum':
            return OpticalPropagation.angular_spectrum(field, distance, wavelength, pixel_pitch)
        else:
            return OpticalPropagation.fresnel_transfer(field, distance, wavelength, pixel_pitch)