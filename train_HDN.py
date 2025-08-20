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
from tests.test_equivariance import *
import torchvision.transforms.functional as TF
import random
import numpy as np


from HeightmapDenoiser import TrivialEquivariantDenoiser
from Utils.dummy_heightmap_dataset import DummyHeightmapDataset
import HeightmapDenoiser as HDN
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor

from tests.test_performance import run_test

MAX_EPOCHS = 100

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
    ):
        super().__init__()
        self.denoising_model = denoising_model
        self.lr = lr
        self.mode = mode
        self.visualize = visualize
        self.visualize_every_n = max(1, visualize_every_n)
        self.num_images_to_log = num_images_to_log

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

        # 1. Main denoising loss (L1 for edge preservation + L2)
        loss_l2 = F.mse_loss(pred, gt)
        # loss_l1 = F.l1_loss(pred, gt)
        # loss_l1 = F.smooth_l1_loss(pred, gt, beta=0.01)
        loss_denoise = loss_l2

        # 3. Robust SSIM structural loss (1 - SSIM, so higher is worse)
        def safe_normalize_for_ssim(t):
            # Ensure tensor is in valid range for SSIM
            t_min, t_max = t.min(), t.max()
            if (t_max - t_min).item() < 1e-6:
                return torch.zeros_like(t)
            return (t - t_min) / (t_max - t_min)

        gt_norm = safe_normalize_for_ssim(gt)
        pred_norm = safe_normalize_for_ssim(pred)
        loss_ssim = 1.0 - ssim(pred_norm, gt_norm, data_range=1.0, size_average=True)

        # Ensure SSIM loss has gradients
        if not loss_ssim.requires_grad:
            loss_ssim = torch.tensor(0.0, device=gt.device, requires_grad=True)

        # 4. Gradient-domain loss (L1 diff between gradients)
        def image_gradients(img):
            dx = img[:, :, :, 1:] - img[:, :, :, :-1]
            dy = img[:, :, 1:, :] - img[:, :, :-1, :]
            return dx, dy

        gt_dx, gt_dy = image_gradients(gt)
        pred_dx, pred_dy = image_gradients(pred)
        loss_grad = (F.l1_loss(pred_dx, gt_dx) + F.l1_loss(pred_dy, gt_dy)) * 0.5

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

        heightmap_gt, heightmap_noised, _ = batch

        # Forward pass with anomaly detection
        try:
            pred = self(heightmap_noised)
            loss, loss_dict  = self.compute_denoising_loss(heightmap_gt, pred, heightmap_noised)
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

        # Optional visualization with Wandb
        if self.visualize and (self.global_step % self.visualize_every_n == 0):
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

    def validation_step(self, batch, batch_idx: int):
        if self.mode != "denoising":
            raise ValueError(f"Invalid mode: {self.mode}")

        heightmap_gt, heightmap_noised, _ = batch

        with torch.no_grad():
            try:
                pred = self(heightmap_noised)
                loss, loss_dict = self.compute_denoising_loss(heightmap_gt, pred, heightmap_noised)
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

        return loss

    def configure_optimizers(self):
        """Configure optimizer with conservative settings for equivariant networks"""
        optimizer = torch.optim.Adam(
            self.denoising_model.parameters(),
            lr=self.lr,
            weight_decay=1e-7,  # Small weight decay for regularization
            eps=1e-8,  # Larger eps for numerical stability
            amsgrad=True  # More stable variant of Adam
        )

        # Optional: Learning rate scheduler
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=MAX_EPOCHS,  # Number of maximum epochs
            eta_min=1e-7  # Minimum learning rate
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
        train_samples: int = 2000,
        val_samples: int = 200,
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
    train_ds = DummyHeightmapDataset(num_samples=train_samples, k=k, noise_std=noise_std, seed=seed)
    val_ds = DummyHeightmapDataset(num_samples=val_samples, k=k, noise_std=noise_std, seed=123)

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
    # denoiser = HDN.AdvancedEquivariantDenoiser(
    #     in_channels=1, num_layers=20, num_feat=128,
    #     max_frequency=1, stability_ratio=0.5)
    denoiser = HDN.TrivialEquivariantDenoiser(num_layers=10, num_feat=128)
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
        "train_samples": train_samples,
        "val_samples": val_samples,
        "num_layers": 10,
        "num_feat": 128,
        "seed": seed,
    })

    # Checkpoint callback (saves to Wandb too with log_model=True)
    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints/denoising",
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
        k=64,
        noise_std=0.08,
        max_epochs=MAX_EPOCHS,
        num_workers=4,
        lr=5e-3,
        visualize=True,
        visualize_every_n=10,  # Log images every 10 steps
        num_images_to_log=5,  # Log 5 example images each time
        log_every_n_steps=5,  # Log metrics every 5 steps (more frequent)
        train_samples = 2000,
        val_samples = 200,
        # Wandb specific settings
        project_name="heightmap-denoising",
        experiment_name="equi-L2+ssim+grad-100epoch",
        wandb_tags=["denoising", "heightmap", "equivariance"],
    )
