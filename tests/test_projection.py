from sympy.printing.pytorch import torch

from HeightmapGenerator import project_points_to_heightmap_exact, project_points_to_heightmap_original
from Utils.dummy_heightmap_dataset import DummyHeightmapDataset, DummyHeightmapDatasetForProjectionTest

import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
import time
from collections import defaultdict


def display_heightmaps(noisy_heightmap, denoised_heightmap, sample_idx=0):
    """Display a single pair of heightmaps for visualization"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Display noisy heightmap
    im1 = axes[0].imshow(noisy_heightmap, cmap='gray')
    axes[0].set_title(f'Original Heightmap (Sample {sample_idx})')
    axes[0].axis('off')
    plt.colorbar(im1, ax=axes[0], shrink=0.8)

    # Display denoised heightmap
    im2 = axes[1].imshow(denoised_heightmap, cmap='gray')
    axes[1].set_title(f'Re-projected Heightmap (Sample {sample_idx})')
    axes[1].axis('off')
    plt.colorbar(im2, ax=axes[2], shrink=0.8)

    plt.tight_layout()
    plt.show()


def calculate_differences(original, reprojected):
    """Calculate various difference metrics between heightmaps"""
    # Ensure both arrays are numpy arrays
    if torch.is_tensor(original):
        original = original.detach().cpu().numpy()
    if torch.is_tensor(reprojected):
        reprojected = reprojected.detach().cpu().numpy()

    # Mean Absolute Error
    mae = np.mean(np.abs(original - reprojected))

    # Mean Squared Error
    mse = np.mean((original - reprojected) ** 2)

    # Root Mean Squared Error
    rmse = np.sqrt(mse)

    # Maximum absolute difference
    max_abs_diff = np.max(np.abs(original - reprojected))

    # Structural Similarity (simplified version)
    mean_orig = np.mean(original)
    mean_reproj = np.mean(reprojected)
    std_orig = np.std(original)
    std_reproj = np.std(reprojected)
    covariance = np.mean((original - mean_orig) * (reprojected - mean_reproj))

    # Correlation coefficient
    correlation = covariance / (std_orig * std_reproj) if (std_orig * std_reproj) > 0 else 0

    return {
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'max_abs_diff': max_abs_diff,
        'correlation': correlation
    }


def plot_difference_statistics(all_differences):
    """Plot histograms of difference metrics across all samples"""
    metrics = ['mae', 'mse', 'rmse', 'max_abs_diff', 'correlation']

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for i, metric in enumerate(metrics):
        values = [diff[metric] for diff in all_differences]
        axes[i].hist(values, bins=30, alpha=0.7, edgecolor='black')
        axes[i].set_title(f'{metric.upper()} Distribution')
        axes[i].set_xlabel(metric.upper())
        axes[i].set_ylabel('Frequency')
        axes[i].grid(True, alpha=0.3)

        # Add statistics text
        mean_val = np.mean(values)
        std_val = np.std(values)
        axes[i].axvline(mean_val, color='red', linestyle='--',
                        label=f'Mean: {mean_val:.4f}')
        axes[i].legend()

    # Remove the extra subplot
    axes[5].remove()

    plt.tight_layout()
    plt.show()


def plot_sample_differences_with_metrics(original, reprojected, sample_idx, rank, mae_val):
    """Plot difference map for worst-performing samples with metrics"""
    difference = original - reprojected

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Original
    im1 = axes[0].imshow(original, cmap='gray')
    axes[0].set_title(f'Original (Sample {sample_idx})')
    axes[0].axis('off')
    plt.colorbar(im1, ax=axes[0], shrink=0.8)

    # Reprojected
    im2 = axes[1].imshow(reprojected, cmap='gray')
    axes[1].set_title(f'Reprojected (Sample {sample_idx})')
    axes[1].axis('off')
    plt.colorbar(im2, ax=axes[1], shrink=0.8)

    # Difference
    max_diff = np.max(np.abs(difference))
    im3 = axes[2].imshow(difference, cmap='RdBu_r', vmin=-max_diff, vmax=max_diff)
    axes[2].set_title(f'Difference (Rank {rank}, MAE: {mae_val:.4f})')
    axes[2].axis('off')
    plt.colorbar(im3, ax=axes[2], shrink=0.8)

    plt.suptitle(f'Worst Sample Analysis - Rank {rank}/10', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()


def plot_sample_differences(original, reprojected, sample_idx, num_display=5):
    """Plot difference map for specific samples"""
    if sample_idx >= num_display:
        return

    difference = original - reprojected

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Original
    im1 = axes[0].imshow(original, cmap='gray')
    axes[0].set_title(f'Original (Sample {sample_idx})')
    axes[0].axis('off')
    plt.colorbar(im1, ax=axes[0], shrink=0.8)

    # Reprojected
    im2 = axes[1].imshow(reprojected, cmap='gray')
    axes[1].set_title(f'Reprojected (Sample {sample_idx})')
    axes[1].axis('off')
    plt.colorbar(im2, ax=axes[1], shrink=0.8)

    # Difference
    im3 = axes[2].imshow(difference, cmap='RdBu_r', vmin=-np.max(np.abs(difference)),
                         vmax=np.max(np.abs(difference)))
    axes[2].set_title(f'Difference (Sample {sample_idx})')
    axes[2].axis('off')
    plt.colorbar(im3, ax=axes[2], shrink=0.8)

    plt.tight_layout()
    plt.show()


def test_projection_1000_samples():
    """Test projection with 1000 samples and measure differences"""
    k = 32
    noise_std = 0.05
    seed = 666
    num_samples = 10

    print(f"Loading {num_samples} samples...")
    test_data = DummyHeightmapDatasetForProjectionTest(
        num_samples=num_samples, k=k, noise_std=noise_std, seed=seed
    )

    all_differences = []
    processing_times = []
    sample_data = []  # Store sample data for later visualization of worst cases

    print("Processing samples and calculating differences...")

    for i in range(num_samples):
        if (i + 1) % 100 == 0:
            print(f"Processed {i + 1}/{num_samples} samples...")

        # Get sample data (getitem() generates a new random sample each time)
        clean_heightmap, noisy_heightmap, per_pixel_normals, normal, noisy_points, clean_points = test_data.getitem()

        # Time the reprojection
        start_time = time.time()
        vertical_norm = torch.tensor([0.0, 0.0, 1.0])
        denoised_heightmap = project_points_to_heightmap_exact(
            [clean_points], [vertical_norm], d_list=None, k=k, r=0.5, sigma=1.0
        )
        processing_time = time.time() - start_time
        processing_times.append(processing_time)

        # Calculate differences between original clean heightmap and reprojected
        differences = calculate_differences(clean_heightmap[0], denoised_heightmap[0])
        all_differences.append(differences)

        # Store sample data for potential visualization later
        sample_data.append({
            'clean': clean_heightmap[0],
            'reprojected': denoised_heightmap[0],
            'sample_idx': i
        })

    # Calculate and display overall statistics
    print("\n" + "=" * 60)
    print("DIFFERENCE STATISTICS ACROSS 1000 SAMPLES")
    print("=" * 60)

    metrics = ['mae', 'mse', 'rmse', 'max_abs_diff', 'correlation']
    for metric in metrics:
        values = [diff[metric] for diff in all_differences]
        mean_val = np.mean(values)
        std_val = np.std(values)
        min_val = np.min(values)
        max_val = np.max(values)

        print(f"{metric.upper()}:")
        print(f"  Mean: {mean_val:.6f} ± {std_val:.6f}")
        print(f"  Range: [{min_val:.6f}, {max_val:.6f}]")
        print()

    # Processing time statistics
    mean_time = np.mean(processing_times)
    total_time = np.sum(processing_times)
    print(f"PROCESSING TIME STATISTICS:")
    print(f"  Mean per sample: {mean_time:.4f} seconds")
    print(f"  Total processing time: {total_time:.2f} seconds")
    print(f"  Samples per second: {num_samples / total_time:.2f}")

    # Plot difference distributions
    plot_difference_statistics(all_differences)

    # Find and display worst cases
    mae_values = [diff['mae'] for diff in all_differences]
    worst_samples = np.argsort(mae_values)[:10]  # 10 worst samples by MAE

    print(f"\nWORST 10 SAMPLES BY MAE:")
    for rank, sample_idx in enumerate(reversed(worst_samples)):
        mae_val = mae_values[sample_idx]
        print(f"  Rank {rank + 1}: Sample {sample_idx}, MAE = {mae_val:.6f}")

    # Plot the worst 10 samples
    print(f"\nPlotting worst 10 samples...")
    for rank, sample_idx in enumerate(reversed(worst_samples)):
        sample = sample_data[sample_idx]
        original_np = sample['clean'].detach().cpu().numpy() if torch.is_tensor(sample['clean']) else sample['clean']
        reprojected_np = sample['reprojected'].detach().cpu().numpy() if torch.is_tensor(sample['reprojected']) else \
        sample['reprojected']
        mae_val = mae_values[sample_idx]

        # Plot with MAE in title
        plot_sample_differences_with_metrics(original_np, reprojected_np, sample_idx, rank + 1, mae_val)

    return all_differences, processing_times


if __name__ == '__main__':
    differences, times = test_projection_1000_samples()