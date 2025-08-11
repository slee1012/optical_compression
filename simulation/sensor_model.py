import torch
import torch.nn as nn


class IdealSensorModel(nn.Module):
    """
    Ideal sensor model that simply outputs the field/intensity at the image plane
    without any processing, noise, digitization, or scaling effects.
    
    This is much cleaner for understanding the core optics behavior.
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
    
    def forward(self, field, add_noise=False):
        """
        Simply return the intensity and field as-is.
        Perfect sensor with no processing artifacts.
        
        Args:
            field: Complex optical field at the image plane
            add_noise: Ignored (ideal sensor has no noise)
            
        Returns:
            dict with 'intensity', 'field', 'snr'
        """
        # Convert to intensity (|field|²)
        intensity = torch.abs(field)**2
        
        return {
            'intensity': intensity,      # Raw intensity (no processing)
            'field': field,             # Original complex field
            'snr': torch.tensor(float('inf')),  # Ideal sensor = infinite SNR
            'noise': torch.zeros_like(intensity)  # No noise
        }


# Keep backward compatibility
class SensorModel(IdealSensorModel):
    """Alias for backward compatibility."""
    pass