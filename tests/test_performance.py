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

from HeightmapDenoiser import TrivialEquivariantDenoiser
from Utils.dummy_heightmap_dataset import DummyHeightmapDataset
from tests.test_equivariance import test_and_plot_equivariance_from_dataset


def run_test(model_path, in_channels, num_layers, num_feat):
    import torch
    import torch.nn.functional as F
    from skimage.metrics import peak_signal_noise_ratio, structural_similarity
    import numpy as np
    from collections import OrderedDict
    checkpoint = torch.load(model_path, map_location='cpu')
    new_state_dict = OrderedDict()
    for k, v in checkpoint['state_dict'].items():
        if k.startswith('denoising_model.'):
            new_state_dict[k[len('denoising_model.'):]] = v
        else:
            new_state_dict[k] = v  # In case there are other keys

    # 3. Load into your model
    model = TrivialEquivariantDenoiser(in_channels=in_channels, num_layers=num_layers, num_feat=num_feat)
    model.load_state_dict(new_state_dict)
    model.eval()

    # 4. Prepare test data
    test_ds = DummyHeightmapDataset(num_samples=100, k=64, noise_std=0.08, seed=666)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)

    import torch
    import torch.nn.functional as F
    from skimage.metrics import peak_signal_noise_ratio, structural_similarity

    def evaluate(model, test_loader):
        model.eval()
        metrics = {
            'l1_loss': 0.0,
            'mse_loss': 0.0,
            'psnr_input': 0.0,
            'psnr_output': 0.0,
            'ssim_input': 0.0,
            'ssim_output': 0.0,
            'mae': 0.0,
        }
        total_samples = 0

        with torch.no_grad():
            for clean_gt, noisy_input, _, _, _ in test_loader:
                pred = model(noisy_input)

                # Ensure tensors are on CPU for skimage metrics
                clean_np = clean_gt.squeeze().cpu().numpy()  # [B, H, W]
                pred_np = pred.squeeze().cpu().numpy()
                noisy_np = noisy_input.squeeze().cpu().numpy()

                # Compute metrics per sample in batch
                batch_size = clean_gt.shape[0]
                total_samples += batch_size

                # L1 Loss (MAE)
                metrics['l1_loss'] += F.l1_loss(pred, clean_gt).item() * batch_size

                # MSE Loss
                metrics['mse_loss'] += F.mse_loss(pred, clean_gt).item() * batch_size

                # PSNR and SSIM (per image)
                for i in range(batch_size):
                    # Input PSNR (noisy vs clean)
                    metrics['psnr_input'] += peak_signal_noise_ratio(
                        clean_np[i], noisy_np[i], data_range=1.0
                    )

                    # Output PSNR (denoised vs clean)
                    metrics['psnr_output'] += peak_signal_noise_ratio(
                        clean_np[i], pred_np[i], data_range=1.0
                    )

                    # SSIM
                    metrics['ssim_input'] += structural_similarity(
                        clean_np[i], noisy_np[i], win_size=7, data_range=1.0
                    )

                    metrics['ssim_output'] += structural_similarity(
                        clean_np[i], pred_np[i], win_size=7, data_range=1.0
                    )

        # Average metrics
        for key in metrics:
            metrics[key] /= total_samples

        print(f"""
            Test Metrics:
            - L1 Loss (MAE): {metrics['l1_loss']:.4f}
            - MSE Loss: {metrics['mse_loss']:.4f}
            - Input PSNR (dB): {metrics['psnr_input']:.2f}
            - Output PSNR (dB): {metrics['psnr_output']:.2f}
            - Input SSIM: {metrics['ssim_input']:.4f}
            - Output SSIM: {metrics['ssim_output']:.4f}
            """)
        return metrics


    evaluate(model, test_loader)
    test_and_plot_equivariance_from_dataset(model, DummyHeightmapDataset)