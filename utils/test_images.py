"""
Utility functions for generating test images for optical simulations.

Provides various 2D test patterns useful for:
- Testing optical systems
- Validating propagation algorithms  
- Analyzing system performance
- Creating consistent test data
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional, Union
from enum import Enum

class ImageType(Enum):
    """Available test image types."""
    GAUSSIAN = "gaussian"
    CIRCLE = "circle"
    SQUARE = "square"
    CROSS = "cross"
    CHECKERBOARD = "checkerboard"
    STRIPES = "stripes"
    SINUSOID = "sinusoid"
    RING = "ring"
    SPOKES = "spokes"
    TEXT = "text"
    RANDOM = "random"
    CAMERAMAN = "cameraman"
    MIXED = "mixed"

def create_test_image(size: Union[int, Tuple[int, int]], 
                     image_type: Union[str, ImageType] = ImageType.GAUSSIAN,
                     **kwargs) -> torch.Tensor:
    """
    Create a test image for optical simulations.
    
    Args:
        size: Image size as (height, width) or single int for square
        image_type: Type of test image to create
        **kwargs: Additional parameters specific to each image type
        
    Returns:
        torch.Tensor: Test image with values in [0, 1]
    """
    if isinstance(size, int):
        size = (size, size)
    
    if isinstance(image_type, str):
        image_type = ImageType(image_type.lower())
    
    # Create coordinate grids
    h, w = size
    y, x = np.meshgrid(np.linspace(-1, 1, h), np.linspace(-1, 1, w), indexing='ij')
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)
    
    # Generate image based on type
    if image_type == ImageType.GAUSSIAN:
        sigma = kwargs.get('sigma', 0.3)
        img = np.exp(-(r**2) / (2 * sigma**2))
        
    elif image_type == ImageType.CIRCLE:
        radius = kwargs.get('radius', 0.4)
        width = kwargs.get('width', 0.05)
        img = np.exp(-((r - radius)**2) / (2 * width**2))
        
    elif image_type == ImageType.SQUARE:
        size_param = kwargs.get('size', 0.5)
        img = ((np.abs(x) < size_param) & (np.abs(y) < size_param)).astype(float)
        
    elif image_type == ImageType.CROSS:
        width = kwargs.get('width', 0.1)
        img = ((np.abs(x) < width) | (np.abs(y) < width)).astype(float)
        
    elif image_type == ImageType.CHECKERBOARD:
        n_squares = kwargs.get('n_squares', 8)
        x_scaled = (x + 1) * n_squares / 2
        y_scaled = (y + 1) * n_squares / 2
        img = ((np.floor(x_scaled) + np.floor(y_scaled)) % 2).astype(float)
        
    elif image_type == ImageType.STRIPES:
        frequency = kwargs.get('frequency', 5)
        direction = kwargs.get('direction', 'vertical')  # 'vertical', 'horizontal', 'diagonal'
        if direction == 'vertical':
            img = 0.5 * (1 + np.sin(frequency * np.pi * x))
        elif direction == 'horizontal':
            img = 0.5 * (1 + np.sin(frequency * np.pi * y))
        else:  # diagonal
            img = 0.5 * (1 + np.sin(frequency * np.pi * (x + y)))
            
    elif image_type == ImageType.SINUSOID:
        freq_x = kwargs.get('freq_x', 3)
        freq_y = kwargs.get('freq_y', 3)
        phase = kwargs.get('phase', 0)
        img = 0.5 * (1 + np.sin(freq_x * np.pi * x + phase) * np.sin(freq_y * np.pi * y + phase))
        
    elif image_type == ImageType.RING:
        inner_radius = kwargs.get('inner_radius', 0.3)
        outer_radius = kwargs.get('outer_radius', 0.5)
        img = ((r >= inner_radius) & (r <= outer_radius)).astype(float)
        
    elif image_type == ImageType.SPOKES:
        n_spokes = kwargs.get('n_spokes', 8)
        spoke_width = kwargs.get('spoke_width', 0.1)
        spoke_angles = np.linspace(0, 2*np.pi, n_spokes, endpoint=False)
        img = np.zeros_like(r)
        for angle in spoke_angles:
            angle_diff = np.abs(np.angle(np.exp(1j * (theta - angle))))
            angle_diff = np.minimum(angle_diff, 2*np.pi - angle_diff)
            img += (angle_diff < spoke_width).astype(float)
        img = np.clip(img, 0, 1)
        
    elif image_type == ImageType.TEXT:
        # Simple "T" pattern
        img = np.zeros_like(r)
        # Horizontal bar of T
        img[h//4:h//4+h//8, w//4:3*w//4] = 1.0
        # Vertical bar of T  
        img[h//4:3*h//4, w//2-w//16:w//2+w//16] = 1.0
        
    elif image_type == ImageType.RANDOM:
        seed = kwargs.get('seed', None)
        if seed is not None:
            np.random.seed(seed)
        img = np.random.rand(h, w)
        
    elif image_type == ImageType.CAMERAMAN:
        # Fallback cameraman-like pattern
        img = _create_cameraman_like(h, w)
        
    elif image_type == ImageType.MIXED:
        # Combination of patterns
        img = 0.3 * np.exp(-(r**2) / (2 * 0.4**2))  # Gaussian background
        img += 0.4 * ((r >= 0.2) & (r <= 0.3)).astype(float)  # Ring
        img += 0.3 * ((np.abs(x) < 0.05) | (np.abs(y) < 0.05)).astype(float)  # Cross
        img = np.clip(img, 0, 1)
        
    else:
        raise ValueError(f"Unknown image type: {image_type}")
    
    # Normalize to [0, 1] and convert to tensor
    img = np.clip(img, 0, 1)
    return torch.tensor(img, dtype=torch.float32)

def _create_cameraman_like(h: int, w: int) -> np.ndarray:
    """Create a cameraman-like test pattern."""
    y, x = np.meshgrid(np.linspace(-1, 1, h), np.linspace(-1, 1, w), indexing='ij')
    
    # Main subject (person-like shape)
    img = np.zeros((h, w))
    
    # Head (circle)
    head_center = (-0.3, 0)
    head_r = np.sqrt((x - head_center[1])**2 + (y - head_center[0])**2)
    img += (head_r < 0.15).astype(float) * 0.8
    
    # Body (rectangle)
    body_mask = ((x >= -0.1) & (x <= 0.1) & (y >= -0.1) & (y <= 0.4))
    img += body_mask.astype(float) * 0.6
    
    # Camera (small rectangle)
    camera_mask = ((x >= 0.0) & (x <= 0.3) & (y >= -0.2) & (y <= 0.0))
    img += camera_mask.astype(float) * 0.9
    
    # Background texture
    img += 0.2 * np.random.rand(h, w)
    
    return np.clip(img, 0, 1)

def create_batch_images(size: Union[int, Tuple[int, int]], 
                       batch_size: int = 4,
                       image_types: Optional[list] = None) -> torch.Tensor:
    """
    Create a batch of different test images.
    
    Args:
        size: Image size
        batch_size: Number of images to create
        image_types: List of image types to use (random if None)
        
    Returns:
        torch.Tensor: Batch of images with shape (batch_size, height, width)
    """
    if image_types is None:
        available_types = [ImageType.GAUSSIAN, ImageType.CIRCLE, ImageType.SQUARE, 
                          ImageType.CROSS, ImageType.CHECKERBOARD, ImageType.SINUSOID,
                          ImageType.RING, ImageType.SPOKES]
        image_types = np.random.choice(available_types, batch_size, replace=True)
    
    if len(image_types) != batch_size:
        image_types = image_types[:batch_size] + image_types * (batch_size // len(image_types) + 1)
        image_types = image_types[:batch_size]
    
    batch = []
    for img_type in image_types:
        img = create_test_image(size, img_type)
        batch.append(img)
    
    return torch.stack(batch, dim=0)

def visualize_test_images(display_grid: bool = True, save_path: Optional[str] = None):
    """
    Create a visualization showing all available test image types.
    
    Args:
        display_grid: Whether to display the grid of images
        save_path: Optional path to save the visualization
    """
    image_types = [ImageType.GAUSSIAN, ImageType.CIRCLE, ImageType.SQUARE, ImageType.CROSS,
                   ImageType.CHECKERBOARD, ImageType.STRIPES, ImageType.SINUSOID, ImageType.RING,
                   ImageType.SPOKES, ImageType.TEXT, ImageType.RANDOM, ImageType.CAMERAMAN, ImageType.MIXED]
    
    n_types = len(image_types)
    cols = 4
    rows = (n_types + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(12, 3 * rows))
    axes = axes.flatten() if n_types > 1 else [axes]
    
    for i, img_type in enumerate(image_types):
        img = create_test_image(64, img_type)
        axes[i].imshow(img.numpy(), cmap='gray')
        axes[i].set_title(f'{img_type.value.title()}')
        axes[i].axis('off')
    
    # Hide unused subplots
    for i in range(n_types, len(axes)):
        axes[i].axis('off')
    
    plt.suptitle('Available Test Image Types', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Test image grid saved to: {save_path}")
    
    if display_grid:
        plt.show()
    else:
        plt.close()
    
    return fig

# Convenience functions for common patterns
def gaussian_beam(size: Union[int, Tuple[int, int]], sigma: float = 0.3) -> torch.Tensor:
    """Create a Gaussian beam pattern."""
    return create_test_image(size, ImageType.GAUSSIAN, sigma=sigma)

def circular_aperture(size: Union[int, Tuple[int, int]], radius: float = 0.4) -> torch.Tensor:
    """Create a circular aperture pattern.""" 
    return create_test_image(size, ImageType.CIRCLE, radius=radius, width=0.02)

def square_aperture(size: Union[int, Tuple[int, int]], aperture_size: float = 0.5) -> torch.Tensor:
    """Create a square aperture pattern."""
    return create_test_image(size, ImageType.SQUARE, size=aperture_size)

def test_target(size: Union[int, Tuple[int, int]]) -> torch.Tensor:
    """Create a test target with multiple features."""
    return create_test_image(size, ImageType.MIXED)

if __name__ == "__main__":
    # Demo the test image generator
    print("Test Image Generator Demo")
    print("=" * 40)
    
    # Create and show sample images
    import os
    os.makedirs('results', exist_ok=True)
    
    # Show all available types
    visualize_test_images(display_grid=False, save_path='results/test_image_types.png')
    
    # Create some specific examples
    gaussian = gaussian_beam(128, sigma=0.2)
    circle = circular_aperture(128, radius=0.3) 
    target = test_target(128)
    
    print(f"Created Gaussian beam: {gaussian.shape}, range [{gaussian.min():.3f}, {gaussian.max():.3f}]")
    print(f"Created circular aperture: {circle.shape}, range [{circle.min():.3f}, {circle.max():.3f}]")
    print(f"Created test target: {target.shape}, range [{target.min():.3f}, {target.max():.3f}]")
    
    print("\\nTest image types visualization saved to: results/test_image_types.png")