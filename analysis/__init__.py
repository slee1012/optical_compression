"""
Analysis module for optical imaging systems.

Contains analysis and evaluation functionality:
- Image quality metrics (PSNR, SSIM, etc.)
- System visualization and plotting
- Performance analysis tools
"""

from .metrics import CompressionMetrics
from .visualization import Visualizer

__all__ = ['CompressionMetrics', 'Visualizer']