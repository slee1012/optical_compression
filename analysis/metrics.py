import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple


class CompressionMetrics:
    
    @staticmethod
    def mse(original, reconstructed):
        return F.mse_loss(reconstructed, original).item()
    
    @staticmethod
    def psnr(original, reconstructed):
        mse = F.mse_loss(reconstructed, original)
        if mse == 0:
            return float('inf')
        max_val = original.max()
        return (20 * torch.log10(max_val / torch.sqrt(mse))).item()
    
    @staticmethod
    def ssim(original, reconstructed, window_size=11, sigma=1.5):
        def create_window(window_size, sigma, device):
            coords = torch.arange(window_size, device=device) - window_size // 2
            g = torch.exp(-(coords**2) / (2 * sigma**2))
            g = g / g.sum()
            window = g.unsqueeze(0) * g.unsqueeze(1)
            window = window / window.sum()
            return window.unsqueeze(0).unsqueeze(0)
        
        device = original.device if hasattr(original, 'device') else torch.device('cpu')
        gaussian = create_window(window_size, sigma, device)
        
        if original.dim() == 2:
            original = original.unsqueeze(0).unsqueeze(0)
            reconstructed = reconstructed.unsqueeze(0).unsqueeze(0)
        elif original.dim() == 3:
            original = original.unsqueeze(1)
            reconstructed = reconstructed.unsqueeze(1)
        
        mu1 = F.conv2d(original, gaussian, padding=window_size//2)
        mu2 = F.conv2d(reconstructed, gaussian, padding=window_size//2)
        
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2
        
        sigma1_sq = F.conv2d(original.pow(2), gaussian, padding=window_size//2) - mu1_sq
        sigma2_sq = F.conv2d(reconstructed.pow(2), gaussian, padding=window_size//2) - mu2_sq
        sigma12 = F.conv2d(original * reconstructed, gaussian, padding=window_size//2) - mu1_mu2
        
        C1 = 0.01**2
        C2 = 0.03**2
        
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        
        return ssim_map.mean().item()
    
    @staticmethod
    def compression_ratio(original_size, compressed_size, original_bits=8, compressed_bits=8):
        original_total = np.prod(original_size) * original_bits
        compressed_total = np.prod(compressed_size) * compressed_bits
        return original_total / compressed_total
    
    @staticmethod
    def frequency_error(original, reconstructed):
        original_fft = torch.fft.fft2(original)
        reconstructed_fft = torch.fft.fft2(reconstructed)
        mag_error = torch.abs(torch.abs(original_fft) - torch.abs(reconstructed_fft))
        return mag_error.mean().item()