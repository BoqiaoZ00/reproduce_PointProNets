import math
import random
from pathlib import Path
from typing import List, Tuple, Optional

import torch
from scipy.ndimage import gaussian_filter
from torch.optim.lr_scheduler import StepLR, ExponentialLR
from torch.utils.data import Dataset
import HeightmapGenerator as HGN
import HeightmapDenoiser as HDN
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from torchvision.transforms import InterpolationMode

import wandb

from Utils import data_loader
from Utils.ground_truth_loader import compute_gt_normals
from Utils.patch_splitter import split_into_patches, split_thimble_into_patches
from main import smooth_heightmap_numpy, plot_heightmap_2d
from tests.test_equivariance import *
import torchvision.transforms.functional as TF
import random
import numpy as np



# Assumed available from your codebase:
# - data_loader.load(folder, device)
# - split_into_patches(vertices, faces, num_patches, patch_radius)
# - compute_gt_normals(vertices, faces) -> (N, 3) or (3,) depending on your impl
# - HGN.project_points_to_heightmap_test(list_of_vertices, list_of_normals, r) -> List[Tensor(k, k)]
# If your signatures differ slightly, tweak in the obvious places below.

class ThimblePatchHeightmapDataset(Dataset):
    """
    Real-data heightmap dataset built from mesh patches.

    Each item returns:
        - heightmap_gt:          FloatTensor (1, k, k)
        - heightmap_noisy:       FloatTensor (1, k, k)
        - per_pixel_normals:     FloatTensor (3, k, k)
        - patch_normal:          FloatTensor (3,)
        - noisy_points:          FloatTensor (N, 3)   (subsampled points in patch coords)

    Notes:
    - We project the *original* patch points onto a local frame defined by patch_normal.
    - The "gt" heightmap is the direct projection result (clipped to [-1.5, 1.5] for consistency).
    - The "noisy" heightmap is obtained by adding Gaussian + occasional spike noise
      to match your synthetic dataset behavior (you can disable this by setting noise_std=0).
    - per-pixel normals are estimated from height gradients (same convention as your Dummy dataset):
          normal ~ [-dx, -dy, 1] then normalized per pixel.
    """

    def __init__(
        self,
        data_path: str = "golf_ball_sim_data/thimble_surrounding_reordered_v_f.obj",
        device: torch.device = torch.device("cpu"),
        k: int = 64,
        r: float = 0.1,
        num_patches_per_mesh: int = 1000,
        patch_radius: float = 0.1,
        subsample_points: int = 100,
        noise_std: float = 0.05,
        salt_pepper_prob: float = 0.3,
        seed: int = 0,
        cache_heightmaps: bool = False,
        cache_dir: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.k = int(k)
        self.r = float(r)
        self.noise_std = float(noise_std)
        self.salt_pepper_prob = float(salt_pepper_prob)
        self.subsample_points = int(subsample_points)
        self.random = random.Random(seed)
        self._torch_gen = torch.Generator(device=device).manual_seed(seed)
        self.device = device

        # load meshes
        self.norm_meshes = data_loader.load_thimble(data_path, device=device)
        print(len(self.norm_meshes[0]))
        print(len(self.norm_meshes[1]))
        # self.norm_meshes = [self.norm_meshes[14]] # only use armadillo for simple runs

        # prepare patch index: a flat list of (mesh_idx, patch_idx)
        self._patches: List[int] = []
        self._patch_vertices: List[torch.Tensor] = []
        self._patch_normals: List[torch.Tensor] = []

        verts = self.norm_meshes[0]
        normals = self.norm_meshes[1]
        patch_lists = split_thimble_into_patches(
            verts, normals,
            num_patches=num_patches_per_mesh,
            patch_radius=patch_radius
        )
        # patch_lists: List[(patch_vertices, patch_normals)]
        for pi, (pverts, pnormals) in enumerate(patch_lists):
            if pverts is None or len(pverts) == 0:
                continue
            self._patches.append(pi)
            self._patch_vertices.append(pverts)  # (P, 3)
            self._patch_normals.append(pnormals)     # (F, 3)

        if len(self._patches) == 0:
            raise RuntimeError("No patches were produced. Check split_into_patches settings.")

        # caching (optional)
        self.cache_heightmaps = cache_heightmaps
        if cache_heightmaps:
            self.cache_dir = Path(cache_dir or "./_heightmap_cache")
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        # precompute coordinate grid for noise
        lin = torch.linspace(-1.0, 1.0, self.k, device=self.device)
        self.yy, self.xx = torch.meshgrid(lin, lin, indexing="ij")  # (k, k)

    def __len__(self) -> int:
        return len(self._patches)

    def __getitem__(self, idx: int):
        pverts = self._patch_vertices[idx]  # tensor (P, 3), device may be cpu
        pnormals = self._patch_normals[idx]     # tensor (F, 3)

        # Ensure on CPU for HGN if needed (or same device as HGN expects)
        pverts_cpu = pverts.detach().cpu()
        pnormals_cpu = pnormals.detach().cpu()

        patch_normal = pnormals_cpu.mean(dim=0)

        # Project patch to heightmap (expects lists)
        clean_heightmaps: List[torch.Tensor] = HGN.project_points_to_heightmap_test(
            [pverts_cpu], [patch_normal], r=self.r, k=self.k
        )
        if len(clean_heightmaps) == 0 or clean_heightmaps[0] is None:
            raise RuntimeError("Projection returned no heightmap.")
        clean = clean_heightmaps[0].to(torch.float32)  # (k, k) expected
        clean_smooth = smooth_heightmap_numpy(clean)

        # MAKE A NOISY *POINT* PATCH, THEN PROJECT AGAIN
        # pverts_noisy = self.add_point_spikes_along_normal(
        #     pverts_cpu, patch_normal, num_points=30, rel_height=0.20, seed=0
        # )
        # # optional small jitter
        # # pverts_noisy = self.add_small_isotropic_jitter(pverts_noisy, sigma_rel_radius=0.003, patch_radius=self.r, seed=0)
        #
        # noisy_heightmaps: List[torch.Tensor] = HGN.project_points_to_heightmap_test(
        #     [pverts_noisy], [patch_normal], r=self.r, k=self.k
        # )
        # if len(noisy_heightmaps) == 0 or noisy_heightmaps[0] is None:
        #     raise RuntimeError("Projection returned no heightmap.")
        # noisy = noisy_heightmaps[0].to(torch.float32)  # (k, k) expected

        # Match your return signature & shapes

        # Treat clean as noise, smoothed clean as the GT
        return (
            clean_smooth.unsqueeze(0).to(torch.float32),                       # (1, k, k)
            clean.unsqueeze(0).to(torch.float32),                       # (1, k, k)
        )

    # ---------- helpers -----------
    @torch.no_grad()
    def smooth_heightmap_numpy(tensor: torch.Tensor, sigma: float = 1.0) -> torch.Tensor:
        # Convert to NumPy
        np_array = tensor.cpu().numpy()
        # Apply Gaussian filter
        smoothed_np = gaussian_filter(np_array, sigma=sigma)
        # Convert back to torch
        return torch.from_numpy(smoothed_np).to(tensor.device)

    @torch.no_grad()
    def add_point_spikes_along_normal(
            self,
            points: torch.Tensor,  # (P,3)
            patch_normal: torch.Tensor,  # (3,)
            *,
            num_points: int = 30,
            rel_height: float = 0.20,  # ~20% of patch z-range
            seed: int = None,
    ) -> torch.Tensor:
        """
        Make sparse 'obvious' outliers by moving random points along the patch normal.
        Returns a new points tensor (P,3) with spikes.
        """
        device, dtype = points.device, points.dtype
        g = torch.Generator(device=device)
        if seed is not None:
            g.manual_seed(seed)

        P = points.size(0)
        if P == 0:
            return points.clone()

        # Estimate the patch’s along-normal height range for scaling
        n = patch_normal / (patch_normal.norm() + 1e-8)
        h = points @ n  # (P,)
        lo, hi = torch.quantile(h, torch.tensor([0.02, 0.98], device=device))
        amp = (hi - lo).clamp(min=1e-6)

        # Which points to spike?
        take = min(num_points, P)
        idx = torch.randperm(P, generator=g, device=device)[:take]

        # Build spikes (± rel_height * amp, with small randomness)
        rand_scale = 0.7 + 0.6 * torch.rand(take, generator=g, device=device)  # ~[0.7,1.3]
        signs = torch.where(torch.rand(take, generator=g, device=device) < 0.5, -1.0, 1.0)
        delta = (rel_height * amp) * rand_scale * signs  # (take,)

        noisy = points.clone()
        noisy[idx] = noisy[idx] + delta.unsqueeze(-1) * n  # move along the normal
        return noisy

    @torch.no_grad()
    def add_small_isotropic_jitter(
            self,
            points: torch.Tensor,
            *,
            sigma_rel_radius: float = 0.003,  # ~0.3% of patch radius (very mild)
            patch_radius: float = 0.15,
            seed: int = None,
    ) -> torch.Tensor:
        """
        Optional gentle jitter on *all* points (simulates sensor noise).
        """
        device, dtype = points.device, points.dtype
        sigma = sigma_rel_radius * patch_radius
        g = torch.Generator(device=device)
        if seed is not None:
            g.manual_seed(seed)
        try:
            # Preferred: randn with explicit size supports generator broadly
            noise = torch.randn(points.shape, device=device, dtype=dtype, generator=g) * sigma
        except TypeError:
            # Very old versions: fall back to manual seeding (global)
            print("fall back to manual seeding (global)")
            cpu = (device.type == "cpu")
            prev_state = torch.random.get_rng_state()
            if cpu:
                torch.manual_seed(seed)
            else:
                torch.cuda.manual_seed_all(seed)
            noise = torch.randn(points.shape, device=device, dtype=dtype) * sigma
            # restore RNG state (best-effort; CPU-only shown)
            torch.random.set_rng_state(prev_state)
        return points + noise


MAX_EPOCHS = 200

class PointProNetDenoise(L.LightningModule):
    """
    Lightning wrapper with proper gradient clipping and NaN handling
    """

    def __init__(
            self,
            denoising_model: torch.nn.Module,
            lr: float = 1e-3,
            mode: str = "denoising",
            visualize: bool = False,
            visualize_every_n: int = 200,
            num_images_to_log: int = 10,
            # Gradient clipping parameters
            gradient_clip_val: float = 0.5,
            gradient_clip_algorithm: str = "norm",
            # NaN detection parameters
            detect_anomaly: bool = True,
            enable_val_rotation: bool = True,
            sweep_val_angles: bool = True,
            val_rotation_seed: int = 1234,
    ):
        super().__init__()
        self.denoising_model = denoising_model
        self.lr = lr
        self.mode = mode
        self.visualize = visualize
        self.visualize_every_n = max(1, visualize_every_n)
        self.num_images_to_log = num_images_to_log

        # Test rotation enabling
        self.enable_val_rotation = enable_val_rotation
        self._val_angles = [a for a in range(45, 360, 45)]  # k for 90°, 180°, 270°
        self._val_rot_rng = random.Random(val_rotation_seed)
        self.sweep_val_angles = sweep_val_angles

        # Gradient clipping settings
        self.gradient_clip_val = gradient_clip_val
        self.gradient_clip_algorithm = gradient_clip_algorithm

        # NaN detection
        self.detect_anomaly = detect_anomaly
        if detect_anomaly:
            torch.autograd.set_detect_anomaly(True)

        # Track NaN occurrences
        self.nan_count = 0
        self.total_steps = 0

        self.save_hyperparameters(ignore=["denoising_model"])

    def configure_gradient_clipping(
            self,
            optimizer,
            gradient_clip_val: Optional[Union[int, float]] = None,
            gradient_clip_algorithm: Optional[str] = None
    ):
        """
        Configure gradient clipping - Lightning will call this automatically
        """
        # Lightning handles the actual clipping, we just need to specify parameters
        self.clip_gradients(
            optimizer,
            gradient_clip_val=self.gradient_clip_val,
            gradient_clip_algorithm=self.gradient_clip_algorithm
        )

        # Optional: Log gradient norms for debugging
        if self.trainer.global_step % 5 == 0:  # Log every 5 steps
            total_norm = 0.0
            param_count = 0
            for p in self.denoising_model.parameters():
                if p.grad is not None:
                    param_norm = p.norm(2)
                    total_norm += param_norm.item() ** 2
                    param_count += 1

            if param_count > 0:
                total_norm = total_norm ** (1. / 2)
                self.log("grad_norm", total_norm, on_step=True, logger=True)

    def forward_with_pad(self, x: torch.Tensor, pad: int = 1) -> torch.Tensor:
        """
        Reflect-pad the input by `pad` pixels, run the model, then crop back
        to the original HxW so targets/masks remain aligned.
        """
        if pad <= 0:
            return self.denoising_model(x)

        # pad = (left, right, top, bottom)
        x_pad = F.pad(x, (pad, pad, pad, pad), mode="reflect")
        y_pad = self.denoising_model(x_pad)

        # Most denoisers preserve spatial size; crop off the padded rim.
        y = y_pad[..., pad:-pad, pad:-pad]
        return y

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.mode != "denoising":
            raise ValueError(f"Invalid mode: {self.mode}")
        return self.denoising_model(x)

    # @staticmethod
    # def compute_denoising_loss(gt: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
    #     return F.mse_loss(gt, pred)

    @staticmethod
    def compute_denoising_loss(
            gt: torch.Tensor,
            pred: torch.Tensor,
            ssim_weight: float = 0.5,
            grad_weight: float = 0.5
    ) -> tuple[torch.Tensor, dict]:
        """
        Compute combined denoising loss with multiple components:
        - L1 pixel loss
        - L1 residual sparsity loss
        - SSIM structural loss
        - Gradient-domain loss

        Args:
            gt: Ground truth clean heightmap
            pred: Model prediction
            ssim_weight: Weight for SSIM structural loss
            grad_weight: Weight for gradient loss

        Returns:
            loss: Combined scalar loss
            loss_dict: Dictionary of individual loss components
        """

        def masked_mse(pred, gt, mask, eps=1e-6):
            w = mask.float()
            return ((pred - gt) ** 2 * w).sum() / w.sum().clamp_min(1.0)

        def masked_grad_l1(pred, gt, mask, eps=1e-6):
            # interior diffs
            dx_p, dy_p = pred[:, :, :, 1:] - pred[:, :, :, :-1], pred[:, :, 1:, :] - pred[:, :, :-1, :]
            dx_g, dy_g = gt[:, :, :, 1:] - gt[:, :, :, :-1], gt[:, :, 1:, :] - gt[:, :, :-1, :]
            # intersect masks where both sides are valid
            mx = (mask[:, :, :, 1:] & mask[:, :, :, :-1]).float()
            my = (mask[:, :, 1:, :] & mask[:, :, :-1, :]).float()
            ldx = (mx * (dx_p - dx_g).abs()).sum() / mx.sum().clamp_min(1.0)
            ldy = (my * (dy_p - dy_g).abs()).sum() / my.sum().clamp_min(1.0)
            return 0.5 * (ldx + ldy)

        def masked_ssim(pred, gt, mask):
            # compute SSIM on the tight bounding box of valid pixels to avoid the frame
            b, _, h, w = mask.shape
            # fallback: erode border by 1 px so SSIM windows stay valid
            import torch.nn.functional as F
            k = 3
            eroded = (1 - F.max_pool2d(1 - mask.float(), k, stride=1, padding=k // 2)).bool()
            # crop to bbox of eroded mask per-batch (simple global crop)
            ys, xs = torch.where(eroded[0, 0])
            if ys.numel() == 0:
                return 0.0 * pred.mean()
            y0, y1, x0, x1 = ys.min().item(), ys.max().item() + 1, xs.min().item(), xs.max().item() + 1
            p = pred[:, :, y0:y1, x0:x1]
            g = gt[:, :, y0:y1, x0:x1]
            p = (p - p.min()) / (p.max() - p.min() + 1e-6)
            g = (g - g.min()) / (g.max() - g.min() + 1e-6)
            return 1.0 - ssim(p, g, data_range=1.0, size_average=True)

        valid = (gt != 0)
        l2 = masked_mse(pred, gt, valid)
        loss_denoise = l2
        loss_ssim = masked_ssim(pred, gt, valid)
        loss_grad = masked_grad_l1(pred, gt, valid)
        loss = l2 + ssim_weight * loss_ssim + grad_weight * loss_grad

        # Check that loss_denoise.requires_grad == Tue
        for name, loss_component in [("denoise", loss_denoise)]:
            if not isinstance(loss_component, torch.Tensor):
                print(f"Warning: {name} loss is not a tensor, converting")
                loss_component = torch.tensor(0.0, device=gt.device, requires_grad=True)
            elif not loss_component.requires_grad:
                print(f" Ignore it if it's in validation process - Warning: {name} loss doesn't require grad")
                loss_component.requires_grad = True

        # Combined weighted loss
        loss = (
                loss_denoise
                + ssim_weight * loss_ssim
                + grad_weight * loss_grad
        )

        loss_dict = {
            "loss_denoise": loss_denoise.detach(),
            "loss_ssim": loss_ssim.detach(),
            "loss_grad": loss_grad.detach(),
            "loss_total": loss.detach(),
        }

        return loss, loss_dict

    @staticmethod
    def compute_additional_metrics(gt: torch.Tensor, pred: torch.Tensor) -> dict:
        """Compute additional regression metrics with NaN protection."""
        with torch.no_grad():
            # Check for NaN/Inf in inputs
            if torch.isnan(gt).any() or torch.isnan(pred).any():
                return {
                    "mae": torch.tensor(float('nan')),
                    "psnr": torch.tensor(float('nan')),
                    "correlation": torch.tensor(float('nan')),
                }

            # Mean Absolute Error
            mae = F.l1_loss(gt, pred)

            # Peak Signal-to-Noise Ratio (PSNR)
            mse = F.mse_loss(gt, pred)
            # Clamp MSE to prevent log(0)
            mse = torch.clamp(mse, min=1e-8)
            psnr = 20 * torch.log10(1.0 / torch.sqrt(mse))

            # Correlation coefficient with numerical stability
            gt_flat = gt.flatten(start_dim=1)
            pred_flat = pred.flatten(start_dim=1)

            gt_mean = gt_flat.mean(dim=1, keepdim=True)
            pred_mean = pred_flat.mean(dim=1, keepdim=True)

            gt_centered = gt_flat - gt_mean
            pred_centered = pred_flat - pred_mean

            # Add epsilon to prevent division by zero
            gt_std = gt_centered.std(dim=1) + 1e-8
            pred_std = pred_centered.std(dim=1) + 1e-8

            correlation = (gt_centered * pred_centered).mean(dim=1) / (gt_std * pred_std)
            correlation = correlation.mean()

            # Clamp correlation to valid range
            correlation = torch.clamp(correlation, -1.0, 1.0)

            return {
                "mae": mae,
                "psnr": psnr,
                "correlation": correlation,
            }

    def _check_for_anomalies(self, loss, pred, batch_idx):
        """Check for NaN/Inf values and log warnings"""
        self.total_steps += 1

        if torch.isnan(loss).any() or torch.isinf(loss).any():
            self.nan_count += 1
            print(f"WARNING: NaN/Inf loss detected at step {self.trainer.global_step}")
            print(f"  Loss: {loss.item() if loss.numel() == 1 else 'multi-element'}")
            print(f"  Pred range: [{pred.min():.4f}, {pred.max():.4f}]")
            print(f"  NaN count: {self.nan_count}/{self.total_steps}")

            # Log to wandb
            if hasattr(self.logger, 'experiment'):
                self.logger.experiment.log({
                    "nan_detection/nan_count": self.nan_count,
                    "nan_detection/nan_rate": self.nan_count / self.total_steps,
                    "nan_detection/step": self.trainer.global_step
                })

            return True
        return False

    def training_step(self, batch, batch_idx: int):
        if self.mode != "denoising":
            raise ValueError(f"Mode not implemented: {self.mode}")

        heightmap_gt, heightmap_noised = batch

        # Forward pass with anomaly detection
        try:
            # pred = self(heightmap_noised)
            pred = self.forward_with_pad(heightmap_noised, pad=1)
            loss, loss_dict  = self.compute_denoising_loss(heightmap_gt, pred)
            # Verify loss has gradients
            if not loss.requires_grad:
                print("WARNING: In the training_step, loss tensor doesn't require gradients!")
                return None
        except RuntimeError as e:
            if "nan" in str(e).lower() or "inf" in str(e).lower():
                print(f"Runtime error with NaN/Inf detected: {e}")
                # Return a dummy loss to continue training
                return torch.tensor(float('nan'), requires_grad=True)
            else:
                raise e

        # Check for anomalies
        has_anomaly = self._check_for_anomalies(loss, pred, batch_idx)

        # If we detect NaN, skip this batch
        if has_anomaly:
            return None  # Lightning will handle this gracefully

        # Log all loss components
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_loss_denoise", loss_dict['loss_denoise'], on_step=True, on_epoch=True)
        self.log("train_loss_ssim", loss_dict['loss_ssim'], on_step=True, on_epoch=True)
        self.log("train_loss_grad", loss_dict['loss_grad'], on_step=True, on_epoch=True)

        # Compute and log additional metrics
        try:
            metrics = self.compute_additional_metrics(heightmap_gt, pred)
            for name, value in metrics.items():
                if not torch.isnan(value):
                    self.log(f"train_{name}", value, on_step=True, on_epoch=True)
        except Exception as e:
            print(f"Error computing metrics: {e}")

        # # Optional visualization with Wandb
        # if self.visualize and (self.global_step % self.visualize_every_n == 0):
        #     with torch.no_grad():
        #         try:
        #             num_examples = min(self.num_images_to_log, heightmap_noised.size(0))
        #             self._log_example_images(
        #                 heightmap_noised[:num_examples],
        #                 heightmap_gt[:num_examples],
        #                 pred[:num_examples],
        #                 num_images=num_examples
        #             )
        #         except Exception as e:
        #             print(f"Error logging images: {e}")
        return loss

    @staticmethod
    def _rotate_batch_degrees(x: torch.Tensor, angle_deg: float) -> torch.Tensor:
        """
        Rotate BCHW tensor by an arbitrary angle in degrees.
        - Bilinear interpolation (keeps things smooth)
        - No expand; keeps size
        - Fills outside with 0 to preserve the “invalid” border convention
        """
        return TF.rotate(
            x, angle=angle_deg,
            interpolation=InterpolationMode.BILINEAR,
            expand=False,
            fill=0.0
        )

    def validation_step(self, batch, batch_idx: int):
        if self.mode != "denoising":
            raise ValueError(f"Invalid mode: {self.mode}")

        heightmap_gt, heightmap_noised = batch

        if self.enable_val_rotation:
            if self.sweep_val_angles:
                print("sweep_val_angles = ", self.sweep_val_angles)
                print("batch_idx = ", batch_idx)
                angle = float(self._val_angles[batch_idx % len(self._val_angles)])
                print("angle = ", angle)
            else:
                angle = float(self._val_rot_rng.choice(self._val_angles))
            # plot_heightmap_2d(heightmap_noised[0][0])
            # plot_heightmap_2d(heightmap_noised[10][0])
            heightmap_noised = self._rotate_batch_degrees(heightmap_noised, angle)
            # plot_heightmap_2d(heightmap_noised[0][0])
            # plot_heightmap_2d(heightmap_noised[10][0])
            heightmap_gt = self._rotate_batch_degrees(heightmap_gt, angle)
        else:
            # angle = 0.0
            heightmap_noised = heightmap_noised
            heightmap_gt = heightmap_gt

        with torch.no_grad():
            try:
                # pred = self(heightmap_noised)
                pred = self.forward_with_pad(heightmap_noised, pad=1)
                loss, loss_dict = self.compute_denoising_loss(heightmap_gt, pred)
            except RuntimeError as e:
                print(f"Validation error: {e}")
                return None

        # Check for anomalies in validation
        if torch.isnan(loss).any() or torch.isinf(loss).any():
            print(f"NaN/Inf in validation at step {self.trainer.global_step}")
            return None

        # Log all loss components
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        self.log("val_loss_denoise", loss_dict['loss_denoise'], on_epoch=True)
        self.log("val_loss_ssim", loss_dict['loss_ssim'], on_epoch=True)
        self.log("val_loss_grad", loss_dict['loss_grad'], on_epoch=True)

        # Compute and log additional metrics
        try:
            metrics = self.compute_additional_metrics(heightmap_gt, pred)
            for name, value in metrics.items():
                if not torch.isnan(value):
                    self.log(f"val_{name}", value, on_step=False, on_epoch=True, prog_bar=True)
        except Exception as e:
            print(f"Error computing validation metrics: {e}")

        # Optional visualization with Wandb
        if self.visualize:  # and (self.global_step % self.visualize_every_n == 0):
            with torch.no_grad():
                try:
                    num_examples = min(self.num_images_to_log, heightmap_noised.size(0))
                    self._log_example_images(
                        heightmap_noised[:num_examples],
                        heightmap_gt[:num_examples],
                        pred[:num_examples],
                        num_images=num_examples
                    )
                except Exception as e:
                    print(f"Error logging images: {e}")

        return loss

    def configure_optimizers(self):
        """Configure optimizer with conservative settings for equivariant networks"""
        optimizer = torch.optim.Adam(
            self.denoising_model.parameters(),
            lr=self.lr,
            weight_decay=1e-8,  # Small weight decay for regularization
            eps=1e-8,  # Larger eps for numerical stability
            amsgrad=True  # More stable variant of Adam
        )

        # Optional: Learning rate scheduler
        # scheduler = CosineAnnealingLR(
        #     optimizer,
        #     T_max=MAX_EPOCHS,  # Number of maximum epochs
        #     eta_min=5e-8  # Minimum learning rate
        # )
        # scheduler = StepLR(
        #     optimizer,
        #     step_size=1,  # Decay the LR every 20 epochs
        #     gamma=0.9  # Multiply LR by 0.5 at each step
        # )

        scheduler = ExponentialLR(
            optimizer,
            gamma=0.95  # The multiplicative factor for decay. Closer to 1.0 = slower decay.
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "frequency": 1,
            }
        }

    def _log_example_images(self, noised: torch.Tensor, gt: torch.Tensor, pred: torch.Tensor, num_images: int = 10):
        """Logs individual noised, ground truth and prediction images to Wandb with NaN checking"""
        if noised.ndim != 4 or noised.size(1) != 1:
            return

        # Check for NaN in images
        if torch.isnan(noised).any() or torch.isnan(gt).any() or torch.isnan(pred).any():
            print("WARNING: NaN detected in images, skipping logging")
            return

        try:
            def prepare_image(t):
                t = t.detach().float()
                # Handle edge case where min == max
                t_min, t_max = t.min(), t.max()
                if t_max - t_min < 1e-8:
                    t = torch.zeros_like(t)
                else:
                    t = (t - t_min) / (t_max - t_min) * 255
                return t.squeeze().cpu().numpy().astype('uint8')

            num_to_log = min(num_images, noised.size(0))
            wandb_images = []

            for i in range(num_to_log):
                try:
                    wandb_images.extend([
                        wandb.Image(prepare_image(noised[i]),
                                    caption=f"Sample {i + 1}: Noised Input"),
                        wandb.Image(prepare_image(gt[i]),
                                    caption=f"Sample {i + 1}: Ground Truth"),
                        wandb.Image(prepare_image(pred[i]),
                                    caption=f"Sample {i + 1}: Prediction")
                    ])
                except Exception as e:
                    print(f"Error preparing image {i}: {e}")
                    continue

            if wandb_images and hasattr(self.logger, 'experiment'):
                self.logger.experiment.log({"examples": wandb_images}, step=self.global_step)

        except Exception as e:
            print(f"Image logging error: {e}")


def run_denoising_training(
        batch_size: int = 32,
        k: int = 64,
        noise_std: float = 0.1,
        max_epochs: int = 30,
        num_workers: int = 4,
        lr: float = 1e-4,
        visualize: bool = True,
        visualize_every_n: int = 10,
        num_images_to_log: int = 5,  # Number of example images to log
        accelerator: Optional[str] = None,
        devices: Optional[int] = None,
        seed: int = 0,
        # Logging control
        log_every_n_steps: int = 5,  # How often to log metrics to Wandb
        # Wandb specific parameters
        project_name: str = "heightmap-denoising",
        experiment_name: Optional[str] = None,
        wandb_tags: Optional[list] = None,
):
    """
    Train the denoiser on a synthetic dummy dataset with Wandb logging.
    """
    L.seed_everything(seed, workers=True)

    # Datasets & loaders
    train_ds = ThimblePatchHeightmapDataset(
        data_path="golf_ball_sim_data/thimble_surrounding_reordered_v_f.obj",
        k=32,              # must match what HGN projector returns
        r=0.1,            # projection radius
        num_patches_per_mesh=2000,
        patch_radius=0.1,
        subsample_points=100,
        noise_std=0.05,    # set to 0.0 if you want the "noisy" = "clean"
        seed=seed
    )

    print("Total training data patches:", len(train_ds))

    val_ds = ThimblePatchHeightmapDataset(
        data_path="golf_ball_sim_data/thimble_surrounding_reordered_v_f.obj",
        k=32,  # must match what HGN projector returns
        r=0.1,  # your projection radius
        num_patches_per_mesh=100,
        patch_radius=0.1,
        subsample_points=100,
        noise_std=0.05,  # set to 0.0 if you want the "noisy" = "clean"
        seed=seed
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
        persistent_workers=(num_workers > 0),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
        persistent_workers=(num_workers > 0),
    )

    # Model
    denoiser = HDN.AdvancedEquivariantDenoiser(in_channels=1, num_layers=10, num_feat=128)
    # denoiser = HDN.AdvancedEquivariantDenoiser(
    #     group="O2",
    #     maximum_frequency=6,  # raise to 8–10 if you need more angular detail
    #     in_channels=1,
    #     base_channels=64,
    #     num_blocks=8,
    #     kernel_size=5,
    #     use_gnorm=True,
    # )
    # denoiser = HDN.HeightmapDenoiser(num_layers=5, num_feat=40)
    # denoiser = HDN.TrivialEquivariantDenoiser(num_layers=10, num_feat=156)
    model = PointProNetDenoise(
        denoising_model=denoiser,
        lr=lr,
        mode="denoising",
        visualize=visualize,
        visualize_every_n=visualize_every_n,
        num_images_to_log=num_images_to_log,
    )

    # Wandb logger setup
    wandb_logger = WandbLogger(
        project=project_name,
        name=experiment_name,
        tags=wandb_tags,
        log_model=True,  # This will upload model checkpoints to Wandb
    )

    # Log hyperparameters to Wandb
    wandb_logger.log_hyperparams({
        "batch_size": batch_size,
        "k": k,
        "noise_std": noise_std,
        "lr": lr,
        "num_layers": 10,
        "num_feat": 128,
        "seed": seed,
    })

    # Checkpoint callback (saves to Wandb too with log_model=True)
    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints/denoising/advanced-10-256/MaxPlanck",
        filename="best-model-{epoch:02d}-{val_loss:.4f}",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        save_last=True,
        verbose=True,
    )

    # Trainer config
    if accelerator is None:
        accelerator = "cpu" #"mps" if torch.backends.mps.is_available() else "cpu"
    if devices is None:
        devices = 1

    lr_monitor = LearningRateMonitor(logging_interval='step')

    trainer = L.Trainer(
        accelerator=accelerator,
        devices=devices,
        max_epochs=max_epochs,
        log_every_n_steps=log_every_n_steps,  # Use the parameter
        logger=wandb_logger,  # Use Wandb instead of TensorBoard
        callbacks=[checkpoint_callback, lr_monitor],
    )

    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    # Log final model artifact to Wandb
    wandb.finish()

    print(f"Best model saved at: {checkpoint_callback.best_model_path}")
    print(f"Wandb run: {wandb_logger.experiment.url}")

    return trainer, checkpoint_callback.best_model_path


if __name__ == "__main__":
    run_denoising_training(
        batch_size=32,
        k=32,
        noise_std=0.08,
        max_epochs=MAX_EPOCHS,
        num_workers=4,
        lr=5e-3,
        visualize=True,
        visualize_every_n=10,  # Log images every 10 steps
        num_images_to_log=5,  # Log 5 example images each time
        log_every_n_steps=5,  # Log metrics every 5 steps (more frequent)

        # Wandb specific settings
        project_name="heightmap-denoising",
        experiment_name="advanced-10-128-chimble-L2-ssim-grad",
        wandb_tags=["denoising", "heightmap", "equivariance", "faces"],
    )

