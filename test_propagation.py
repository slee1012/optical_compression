"""
Test propagation module with simple binary patterns and converging wavefronts.
This script helps verify physical behavior of optical wave propagation.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from core.propagation import OpticalPropagation
import warnings
warnings.filterwarnings('ignore')

def create_simple_patterns(resolution=(256, 256)):
    """Create simple binary test patterns for propagation testing."""
    patterns = {}
    
    # Single slit
    slit = torch.zeros(resolution)
    slit[:, resolution[1]//2-5:resolution[1]//2+5] = 1
    patterns['single_slit'] = slit
    
    # Double slit
    double_slit = torch.zeros(resolution)
    double_slit[:, resolution[1]//2-25:resolution[1]//2-15] = 1
    double_slit[:, resolution[1]//2+15:resolution[1]//2+25] = 1
    patterns['double_slit'] = double_slit
    
    # Square aperture
    square = torch.zeros(resolution)
    center_y, center_x = resolution[0]//2, resolution[1]//2
    square[center_y-20:center_y+20, center_x-20:center_x+20] = 1
    patterns['square'] = square
    
    # Circular aperture
    circular = torch.zeros(resolution)
    y, x = torch.meshgrid(torch.arange(resolution[0]), torch.arange(resolution[1]), indexing='ij')
    radius = 30
    mask = ((x - center_x)**2 + (y - center_y)**2) <= radius**2
    circular[mask] = 1
    patterns['circular'] = circular
    
    # Cross pattern
    cross = torch.zeros(resolution)
    cross[resolution[0]//2-3:resolution[0]//2+3, :] = 1
    cross[:, resolution[1]//2-3:resolution[1]//2+3] = 1
    patterns['cross'] = cross
    
    return patterns

def create_converging_wavefront(resolution=(256, 256), focal_length=1e-3, wavelength=550e-9, pixel_size=3.45e-6):
    """Create a converging spherical wavefront using quadratic phase."""
    y, x = torch.meshgrid(
        torch.arange(resolution[0], dtype=torch.float32),
        torch.arange(resolution[1], dtype=torch.float32),
        indexing='ij'
    )
    
    # Center coordinates
    y = (y - resolution[0]/2) * pixel_size
    x = (x - resolution[1]/2) * pixel_size
    
    # Quadratic phase for convergence (acts like a lens)
    k = 2 * np.pi / wavelength
    r_squared = x**2 + y**2
    
    # Phase delay for converging wavefront (negative for convergence)
    phase = -k * r_squared / (2 * focal_length)
    
    # Create complex field with uniform amplitude and converging phase
    field = torch.complex(torch.ones(resolution), torch.zeros(resolution))
    field = field * torch.exp(1j * phase)
    
    # Add circular aperture to limit the beam
    radius_pixels = 40
    y_idx, x_idx = torch.meshgrid(
        torch.arange(resolution[0]), 
        torch.arange(resolution[1]), 
        indexing='ij'
    )
    mask = ((x_idx - resolution[1]/2)**2 + (y_idx - resolution[0]/2)**2) <= radius_pixels**2
    field = field * mask.float()
    
    return field

def test_propagation_distances(pattern, wavelength=550e-9, pixel_size=3.45e-6):
    """Test propagation at multiple distances."""
    # Convert pattern to complex field
    field = torch.complex(pattern.float(), torch.zeros_like(pattern))
    
    # Test distances (in mm)
    distances_mm = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
    distances = [d * 1e-3 for d in distances_mm]  # Convert to meters
    
    results = []
    for d in distances:
        propagated = OpticalPropagation.angular_spectrum(field, d, wavelength, pixel_size)
        results.append({
            'distance_mm': d * 1000,
            'field': propagated,
            'intensity': torch.abs(propagated)**2,
            'phase': torch.angle(propagated)
        })
    
    return results

def test_converging_propagation(focal_length=1e-3, wavelength=550e-9, pixel_size=3.45e-6):
    """Test converging wavefront propagation around focal point."""
    # Create converging wavefront
    converging_field = create_converging_wavefront(
        resolution=(256, 256),
        focal_length=focal_length,
        wavelength=wavelength,
        pixel_size=pixel_size
    )
    
    # Test around focal point (0.5mm to 1.5mm)
    distances_mm = [0.5, 0.7, 0.9, 1.0, 1.1, 1.3, 1.5]
    distances = [d * 1e-3 for d in distances_mm]
    
    results = []
    for d in distances:
        propagated = OpticalPropagation.angular_spectrum(converging_field, d, wavelength, pixel_size)
        results.append({
            'distance_mm': d * 1000,
            'field': propagated,
            'intensity': torch.abs(propagated)**2,
            'phase': torch.angle(propagated)
        })
    
    return results

def plot_propagation_results(pattern_name, results, save_path=None):
    """Plot propagation results at different distances."""
    n_distances = len(results)
    fig = plt.figure(figsize=(16, 8))
    gs = GridSpec(2, n_distances, hspace=0.3, wspace=0.3)
    
    # Find global min/max for consistent colorscale
    all_intensities = torch.stack([r['intensity'] for r in results])
    vmax = all_intensities.max().item()
    
    for i, result in enumerate(results):
        # Intensity plot
        ax1 = fig.add_subplot(gs[0, i])
        im1 = ax1.imshow(result['intensity'].numpy(), cmap='hot', vmin=0, vmax=vmax)
        ax1.set_title(f'{result["distance_mm"]:.1f}mm')
        ax1.axis('off')
        if i == n_distances - 1:
            plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
        
        # Phase plot
        ax2 = fig.add_subplot(gs[1, i])
        im2 = ax2.imshow(result['phase'].numpy(), cmap='twilight', vmin=-np.pi, vmax=np.pi)
        ax2.set_title('Phase')
        ax2.axis('off')
        if i == n_distances - 1:
            plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    
    fig.suptitle(f'Propagation of {pattern_name}', fontsize=16)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.savefig(f'results/{pattern_name.lower().replace(" ", "_")}_propagation.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    return fig

def plot_converging_results(results, focal_length_mm=1.0):
    """Plot converging wavefront propagation around focal point."""
    n_distances = len(results)
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(3, n_distances, hspace=0.3, wspace=0.3)
    
    # Find global min/max
    all_intensities = torch.stack([r['intensity'] for r in results])
    vmax = all_intensities.max().item()
    
    # Store metrics for analysis
    spot_sizes = []
    peak_intensities = []
    
    for i, result in enumerate(results):
        # Intensity plot
        ax1 = fig.add_subplot(gs[0, i])
        im1 = ax1.imshow(result['intensity'].numpy(), cmap='hot', vmin=0, vmax=vmax)
        dist = result["distance_mm"]
        ax1.set_title(f'{dist:.1f}mm' + (' (focal)' if abs(dist - focal_length_mm) < 0.01 else ''))
        ax1.axis('off')
        
        # Phase plot
        ax2 = fig.add_subplot(gs[1, i])
        im2 = ax2.imshow(result['phase'].numpy(), cmap='twilight', vmin=-np.pi, vmax=np.pi)
        ax2.axis('off')
        
        # Cross-section plot
        ax3 = fig.add_subplot(gs[2, i])
        center_idx = result['intensity'].shape[0] // 2
        cross_section = result['intensity'][center_idx, :].numpy()
        ax3.plot(cross_section)
        ax3.set_ylim([0, vmax])
        ax3.set_title('Cross-section')
        ax3.grid(True, alpha=0.3)
        
        # Calculate metrics
        peak_intensities.append(result['intensity'].max().item())
        # Estimate spot size (FWHM)
        threshold = result['intensity'].max() / 2
        above_threshold = (result['intensity'] > threshold).float()
        spot_sizes.append(above_threshold.sum().item())
    
    fig.suptitle(f'Converging Wavefront (focal length = {focal_length_mm:.1f}mm)', fontsize=16)
    
    # Add colorbar
    fig.colorbar(im1, ax=fig.get_axes()[:n_distances], fraction=0.02, pad=0.04, label='Intensity')
    fig.colorbar(im2, ax=fig.get_axes()[n_distances:2*n_distances], fraction=0.02, pad=0.04, label='Phase')
    
    plt.savefig(f'results/converging_wavefront_propagation.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Plot metrics
    fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    distances = [r['distance_mm'] for r in results]
    ax1.plot(distances, peak_intensities, 'o-')
    ax1.axvline(focal_length_mm, color='r', linestyle='--', label='Expected focal point')
    ax1.set_xlabel('Distance (mm)')
    ax1.set_ylabel('Peak Intensity')
    ax1.set_title('Peak Intensity vs Distance')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    ax2.plot(distances, spot_sizes, 'o-')
    ax2.axvline(focal_length_mm, color='r', linestyle='--', label='Expected focal point')
    ax2.set_xlabel('Distance (mm)')
    ax2.set_ylabel('Spot Size (pixels)')
    ax2.set_title('Spot Size vs Distance')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(f'results/converging_wavefront_metrics.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    return fig, fig2

def compare_propagation_methods(pattern, distance=1e-3, wavelength=550e-9, pixel_size=3.45e-6):
    """Compare Angular Spectrum and Fresnel propagation methods."""
    # Propagate
    field = torch.complex(pattern.float(), torch.zeros_like(pattern))
    
    result_as = OpticalPropagation.angular_spectrum(field, distance, wavelength, pixel_size)
    result_fresnel = OpticalPropagation.fresnel_transfer(field, distance, wavelength, pixel_size)
    
    # Plot comparison
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Angular Spectrum
    axes[0, 0].imshow(torch.abs(result_as)**2, cmap='hot')
    axes[0, 0].set_title('Angular Spectrum - Intensity')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(torch.angle(result_as), cmap='twilight', vmin=-np.pi, vmax=np.pi)
    axes[0, 1].set_title('Angular Spectrum - Phase')
    axes[0, 1].axis('off')
    
    # Fresnel
    axes[1, 0].imshow(torch.abs(result_fresnel)**2, cmap='hot')
    axes[1, 0].set_title('Fresnel - Intensity')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(torch.angle(result_fresnel), cmap='twilight', vmin=-np.pi, vmax=np.pi)
    axes[1, 1].set_title('Fresnel - Phase')
    axes[1, 1].axis('off')
    
    # Difference
    diff = torch.abs(result_as - result_fresnel)**2
    im = axes[0, 2].imshow(diff, cmap='coolwarm')
    axes[0, 2].set_title('Intensity Difference')
    axes[0, 2].axis('off')
    plt.colorbar(im, ax=axes[0, 2])
    
    # Cross-sections
    center = result_as.shape[0] // 2
    axes[1, 2].plot(torch.abs(result_as[center, :])**2, label='Angular Spectrum')
    axes[1, 2].plot(torch.abs(result_fresnel[center, :])**2, label='Fresnel', linestyle='--')
    axes[1, 2].set_title('Center Cross-section')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)
    
    fig.suptitle(f'Method Comparison at {distance*1000:.1f}mm', fontsize=16)
    plt.tight_layout()
    plt.savefig(f'results/method_comparison_{int(distance*1000)}mm.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    return fig

def main():
    print("=" * 80)
    print("OPTICAL PROPAGATION TESTING")
    print("=" * 80)
    
    # Create test patterns
    print("\n1. Creating test patterns...")
    patterns = create_simple_patterns()
    print(f"   Created {len(patterns)} test patterns")
    
    # Test 1: Single slit diffraction
    print("\n2. Testing single slit diffraction pattern...")
    results_slit = test_propagation_distances(patterns['single_slit'])
    plot_propagation_results('Single Slit', results_slit)
    
    # Test 2: Double slit interference
    print("\n3. Testing double slit interference pattern...")
    results_double = test_propagation_distances(patterns['double_slit'])
    plot_propagation_results('Double Slit', results_double)
    
    # Test 3: Circular aperture diffraction
    print("\n4. Testing circular aperture (Airy pattern)...")
    results_circular = test_propagation_distances(patterns['circular'])
    plot_propagation_results('Circular Aperture', results_circular)
    
    # Test 4: Square aperture
    print("\n5. Testing square aperture diffraction...")
    results_square = test_propagation_distances(patterns['square'])
    plot_propagation_results('Square Aperture', results_square)
    
    # Test 5: Converging wavefront
    print("\n6. Testing converging wavefront (focal length = 1mm)...")
    results_converging = test_converging_propagation(focal_length=1e-3)
    plot_converging_results(results_converging, focal_length_mm=1.0)
    
    # Test 6: Method comparison
    print("\n7. Comparing Angular Spectrum vs Fresnel methods...")
    compare_propagation_methods(patterns['double_slit'], distance=2e-3)
    
    print("\n" + "=" * 80)
    print("PHYSICAL INSIGHTS:")
    print("-" * 80)
    print("1. Single/Double Slit: Should show classical diffraction/interference patterns")
    print("2. Circular Aperture: Should show Airy disk pattern with concentric rings")
    print("3. Square Aperture: Should show square-symmetric diffraction pattern")
    print("4. Converging Wavefront: Should focus to minimum spot at focal distance")
    print("5. Near-field (<1mm): Fresnel diffraction regime, complex patterns")
    print("6. Far-field (>5mm): Fraunhofer diffraction, cleaner patterns")
    print("=" * 80)

if __name__ == "__main__":
    main()