from sympy.printing.pytorch import torch

from HeightmapGenerator import project_points_to_heightmap_exact, project_points_to_heightmap_original
from Utils.dummy_heightmap_dataset import DummyHeightmapDataset, DummyHeightmapDatasetForProjectionTest

import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F


def display_heightmaps(noisy_heightmap, denoised_heightmap):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Display noisy heightmap
    im1 = axes[0].imshow(noisy_heightmap, cmap='gray')
    axes[0].set_title('Noisy Heightmap')
    axes[0].axis('off')
    plt.colorbar(im1, ax=axes[0], shrink=0.8)

    # Display denoised heightmap
    im2 = axes[1].imshow(denoised_heightmap, cmap='gray')
    axes[1].set_title('Denoised Heightmap')
    axes[1].axis('off')
    plt.colorbar(im2, ax=axes[1], shrink=0.8)

    plt.tight_layout()
    plt.show()

def test_projection():
    k = 32
    noise_std = 0.05
    seed = 10
    test_data = DummyHeightmapDatasetForProjectionTest(num_samples=1, k=k, noise_std=noise_std, seed=seed)
    clean_heightmap, noisy_heightmap, per_pixel_normals, normal, noisy_points, clean_points = test_data.getitem()

    d_x = torch.tensor([0.0, 1.0, 0.0], dtype=normal.dtype)
    d_x = d_x - torch.dot(d_x, normal) * normal
    d_x = F.normalize(d_x, dim=0)

    denoised_heightmap = project_points_to_heightmap_exact(
        [clean_points], [normal], d_list=None, k=k, r=1.0, sigma=1.0
    )
    display_heightmaps(clean_heightmap[0], denoised_heightmap[0])


if __name__ == '__main__':
    test_projection()