import torch
import torch.nn as nn
import torch.nn.functional as F


class SensorModel(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.read_noise = config.readout_noise
        self.bit_depth = config.bit_depth
        self.full_well = 2**config.bit_depth - 1
        self.subsample_factor = config.subsample_factor
        self.quantum_efficiency = 0.8
        self.dark_current = 0.1
        self.integration_time = 0.01
    
    def forward(self, field, add_noise=False):
        intensity = torch.abs(field)**2
        
        max_photons = self.full_well / self.quantum_efficiency
        intensity = intensity / (intensity.max() + 1e-8) * max_photons
        
        electrons = intensity * self.quantum_efficiency
        dark_electrons = self.dark_current * self.integration_time
        electrons = electrons + dark_electrons
        
        if add_noise and not self.training:
            shot_noise = torch.sqrt(electrons) * torch.randn_like(electrons)
            read_noise = self.read_noise * torch.randn_like(electrons)
            electrons = electrons + shot_noise + read_noise
            electrons = torch.clamp(electrons, min=0)
        
        if self.subsample_factor > 1:
            electrons = F.avg_pool2d(
                electrons.unsqueeze(1) if electrons.dim() == 2 else electrons.unsqueeze(0).unsqueeze(0),
                kernel_size=self.subsample_factor,
                stride=self.subsample_factor
            ).squeeze()
        
        digital_output = torch.round(electrons * self.full_well / electrons.max())
        digital_output = torch.clamp(digital_output, 0, self.full_well)
        
        signal = electrons.mean()
        noise_var = self.read_noise**2 + signal
        snr = 10 * torch.log10(signal**2 / (noise_var + 1e-8))
        
        return {
            'intensity': digital_output / self.full_well,
            'electrons': electrons,
            'snr': snr,
            'noise': torch.sqrt(noise_var)
        }