import torch
import torch.nn.functional as F
from PIL import Image
from matplotlib import pyplot as plt
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
import lightning as L
from typing import Optional, Union
import wandb
from pytorch_msssim import ssim
import torchvision.transforms.functional as TF
import random
import numpy as np
import torch
import torch.nn.functional as F
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import numpy as np
from collections import OrderedDict
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import rotate


def test_and_plot_equivariance_from_dataset(model, dataset_class):
    """
    Tests if a model is rotationally equivariant by using a synthetic dataset
    and plotting the results.

    Args:
        model (torch.nn.Module): The equivariant model to test.
        dataset_class (type): The class of the dataset (e.g., DummyHeightmapDataset).
    """

    try:
        # 1. Create a single sample from the dataset
        dummy_ds = dataset_class(num_samples=1, seed=999)
        heightmap_gt, heightmap_noisy, normal_support_placeholder = dummy_ds[0]
        input_tensor = heightmap_noisy  # Use the noisy data for testing

        # Add a batch dimension if necessary
        if input_tensor.dim() == 3:
            input_tensor = input_tensor.unsqueeze(0)

        # Set the model to evaluation mode
        model.eval()

        # 2. Generate a rotation angle (use multiples of 90° for exact rotations)
        angle = np.random.choice([90, 180, 270])  # Exact rotations to avoid interpolation errors

        # 3. Rotate the input tensor using scipy (more precise than PIL)
        input_numpy = input_tensor.squeeze().numpy()
        rotated_input_numpy = rotate(input_numpy, angle, axes=(0, 1), reshape=False, order=1, prefilter=False)
        rotated_input_tensor = torch.from_numpy(rotated_input_numpy).float().unsqueeze(0).unsqueeze(0)

        # 4. Get the model's output for both inputs
        with torch.no_grad():
            output_original = model(input_tensor)
            output_rotated_input = model(rotated_input_tensor)

        # 5. Rotate the original output by the same angle (this is what equivariance predicts)
        output_original_numpy = output_original.squeeze().numpy()
        rotated_original_output_numpy = rotate(output_original_numpy, angle, axes=(0, 1), reshape=False, order=1,
                                               prefilter=False)
        rotated_original_output_tensor = torch.from_numpy(rotated_original_output_numpy).float().unsqueeze(0).unsqueeze(
            0)

        # 6. Compare: f(R(x)) should equal R(f(x))
        # Crop to avoid edge effects from rotation
        crop_size = min(output_rotated_input.shape[-2:]) // 4
        h_start, w_start = crop_size, crop_size
        h_end, w_end = -crop_size if crop_size > 0 else None, -crop_size if crop_size > 0 else None

        output_rotated_cropped = output_rotated_input[:, :, h_start:h_end, w_start:w_end]
        rotated_output_cropped = rotated_original_output_tensor[:, :, h_start:h_end, w_start:w_end]

        diff = torch.abs(output_rotated_cropped - rotated_output_cropped)
        max_diff = diff.max()
        mean_diff = diff.mean()

        tolerance = 1e-3  # More realistic tolerance for floating point operations

        if max_diff < tolerance:
            print(f"Equivariance test PASSED! Max difference: {max_diff:.6f}, Mean difference: {mean_diff:.6f}")
        else:
            print(f"Equivariance test FAILED! Max difference: {max_diff:.6f}, Mean difference: {mean_diff:.6f}")

        # 7. Plot the images for visual inspection
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle(f'Equivariance Test (Rotation: {angle} degrees)', fontsize=16)

        # Set the correct data range based on your dataset's clamping
        vmin, vmax = -1.5, 1.5

        # Top row: inputs
        axes[0, 0].imshow(input_tensor.squeeze().numpy(), cmap='gray', vmin=vmin, vmax=vmax)
        axes[0, 0].set_title("Original Input")
        axes[0, 0].axis('off')

        axes[0, 1].imshow(rotated_input_tensor.squeeze().numpy(), cmap='gray', vmin=vmin, vmax=vmax)
        axes[0, 1].set_title(f"Rotated Input ({angle}°)")
        axes[0, 1].axis('off')

        axes[0, 2].imshow(heightmap_gt.squeeze().numpy(), cmap='gray')
        axes[0, 2].set_title(f"Ground Truth")
        axes[0, 2].axis('off')

        # Bottom row: outputs
        axes[1, 0].imshow(output_original.squeeze().numpy(), cmap='gray', vmin=vmin, vmax=vmax)
        axes[1, 0].set_title("f(Original Input)")
        axes[1, 0].axis('off')

        axes[1, 1].imshow(output_rotated_input.squeeze().numpy(), cmap='gray', vmin=vmin, vmax=vmax)
        axes[1, 1].set_title("f(Rotated Input)")
        axes[1, 1].axis('off')

        axes[1, 2].imshow(rotated_original_output_tensor.squeeze().numpy(), cmap='gray', vmin=vmin, vmax=vmax)
        axes[1, 2].set_title("Rotated f(Original)")
        axes[1, 2].axis('off')

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

        return max_diff.item(), mean_diff.item()

    except Exception as e:
        print(f"An error occurred: {e}")
        return None, None


def test_equivariance_multiple_angles(model, dataset_class, angles=[90, 180, 270]):
    """
    Test equivariance across multiple rotation angles.
    """
    print("Testing equivariance across multiple angles:")
    print("-" * 50)

    results = []
    for angle in angles:
        print(f"\nTesting {angle}° rotation:")

        try:
            dummy_ds = dataset_class(num_samples=1, seed=123)
            heightmap_gt, heightmap_noisy, normal_support_placeholder = dummy_ds[0]
            input_tensor = heightmap_noisy  # Use the noisy data for testing

            if input_tensor.dim() == 3:
                input_tensor = input_tensor.unsqueeze(0)

            model.eval()

            # Rotate input
            input_numpy = input_tensor.squeeze().numpy()
            rotated_input_numpy = rotate(input_numpy, angle, axes=(0, 1), reshape=False, order=1, prefilter=False)
            rotated_input_tensor = torch.from_numpy(rotated_input_numpy).float().unsqueeze(0).unsqueeze(0)

            # Get outputs
            with torch.no_grad():
                output_original = model(input_tensor)
                output_rotated_input = model(rotated_input_tensor)

            # Rotate original output
            output_original_numpy = output_original.squeeze().numpy()
            rotated_original_output_numpy = rotate(output_original_numpy, angle, axes=(0, 1), reshape=False, order=1,
                                                   prefilter=False)
            rotated_original_output_tensor = torch.from_numpy(rotated_original_output_numpy).float().unsqueeze(
                0).unsqueeze(0)

            # Compare with cropping
            crop_size = min(output_rotated_input.shape[-2:]) // 4
            if crop_size > 0:
                h_start, w_start = crop_size, crop_size
                h_end, w_end = -crop_size, -crop_size
                output_rotated_cropped = output_rotated_input[:, :, h_start:h_end, w_start:w_end]
                rotated_output_cropped = rotated_original_output_tensor[:, :, h_start:h_end, w_start:w_end]
            else:
                output_rotated_cropped = output_rotated_input
                rotated_output_cropped = rotated_original_output_tensor

            diff = torch.abs(output_rotated_cropped - rotated_output_cropped)
            max_diff = diff.max().item()
            mean_diff = diff.mean().item()

            results.append((angle, max_diff, mean_diff))

            status = "PASSED" if max_diff < 1e-3 else "FAILED"
            print(f"  {status}: Max diff = {max_diff:.6f}, Mean diff = {mean_diff:.6f}")

        except Exception as e:
            print(f"  ERROR: {e}")
            results.append((angle, float('inf'), float('inf')))

    print("\n" + "=" * 50)
    print("SUMMARY:")
    overall_passed = all(max_diff < 1e-3 for _, max_diff, _ in results if max_diff != float('inf'))
    print(f"Overall equivariance test: {'PASSED' if overall_passed else 'FAILED'}")

    return results