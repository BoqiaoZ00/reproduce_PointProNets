import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import lightning as L
from typing import Optional

from Utils.dummy_heightmap_dataset import DummyHeightmapDataset
import HeightmapDenoiser as HDN
from lightning.pytorch.loggers import TensorBoardLogger

class PointProNet(L.LightningModule):
    """
    Lightning wrapper for training a heightmap denoiser (and later, normals if needed).
    """

    def __init__(
        self,
        denoising_model: torch.nn.Module,
        lr: float = 1e-3,
        mode: str = "denoising",
        visualize: bool = False,
        visualize_every_n: int = 200,
    ):
        super().__init__()
        self.denoising_model = denoising_model
        self.lr = lr
        self.mode = mode
        self.visualize = visualize
        self.visualize_every_n = max(1, visualize_every_n)

        self.save_hyperparameters(ignore=["denoising_model"])

    # -------- Core --------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.mode != "denoising":
            raise ValueError(f"Invalid mode: {self.mode}")
        return self.denoising_model(x)

    @staticmethod
    def compute_denoising_loss(gt: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(gt, pred)

    # -------- Lightning hooks --------
    def training_step(self, batch, batch_idx: int):
        if self.mode != "denoising":
            raise ValueError(f"Mode not implemented: {self.mode}")

        heightmap_gt, heightmap_noised, _ = batch
        pred = self(heightmap_noised)
        loss = self.compute_denoising_loss(heightmap_gt, pred)

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)

        # Optional lightweight visualization (logs as tensors; doesn't block)
        if self.visualize and (self.global_step % self.visualize_every_n == 0):
            with torch.no_grad():
                # log first item of the batch
                self._log_example_images(heightmap_noised[0:1], heightmap_gt[0:1], pred[0:1])

        return loss

    def validation_step(self, batch, batch_idx: int):
        if self.mode != "denoising":
            raise ValueError(f"Invalid mode: {self.mode}")

        heightmap_gt, heightmap_noised, _ = batch
        with torch.no_grad():
            pred = self(heightmap_noised)
            loss = self.compute_denoising_loss(heightmap_gt, pred)

        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.denoising_model.parameters(), lr=self.lr)

    # -------- Helpers --------
    def _log_example_images(self, noised: torch.Tensor, gt: torch.Tensor, pred: torch.Tensor):
        """
        Logs a small stack of example images as a single Hx(W*3) grid to the logger (if supported).
        Falls back to doing nothing if logger doesn't support image logging.
        """
        # Expect shape [1, 1, H, W]
        if noised.ndim != 4 or noised.size(1) != 1:
            return

        try:
            # Normalize to [0,1] for viewing
            def norm01(t):
                t = t.detach().float()
                t = (t - t.min()) / (t.max() - t.min() + 1e-8)
                return t

            n, g, p = map(norm01, (noised, gt, pred))  # [1,1,H,W]
            grid = torch.cat([n[0], g[0], p[0]], dim=-1)  # [1,H, 3W]
            self.logger.experiment.add_image("examples/noised|gt|pred", grid, self.global_step)
        except Exception:
            # If logger doesn't support add_image, just skip silently.
            pass


def run_denoising_training(
    batch_size: int = 16,
    k: int = 64,
    noise_std: float = 0.08,
    max_epochs: int = 5,
    num_workers: int = 0,
    lr: float = 1e-3,
    visualize: bool = True,
    visualize_every_n: int = 200,
    accelerator: Optional[str] = None,
    devices: Optional[int] = None,
    train_samples: int = 2000,
    val_samples: int = 200,
    seed: int = 0,
):
    """
    Train the denoiser on a synthetic dummy dataset.
    """
    L.seed_everything(seed, workers=True)

    # Datasets & loaders
    train_ds = DummyHeightmapDataset(num_samples=train_samples, k=k, noise_std=noise_std, seed=seed)
    val_ds   = DummyHeightmapDataset(num_samples=val_samples, k=k, noise_std=noise_std, seed=123)

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
    denoiser = HDN.HeightmapDenoiser(in_channels=1, num_layers=10, num_feat=64)
    model = PointProNet(
        denoising_model=denoiser,
        lr=lr,
        mode="denoising",
        visualize=visualize,
        visualize_every_n=visualize_every_n,
    )

    # Trainer config
    if accelerator is None:
        accelerator = "cpu" #"gpu" if torch.cuda.is_available() else ("mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu")
    if devices is None:
        devices = 1

    trainer = L.Trainer(
        accelerator=accelerator,
        devices=devices,
        max_epochs=max_epochs,
        log_every_n_steps=10,
        logger=TensorBoardLogger(save_dir="logs_denoising"),
    )
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)


if __name__ == "__main__":
    run_denoising_training(
        batch_size=32,
        k=64,
        noise_std=0.08,
        max_epochs=50,
        num_workers=4,
        lr=1e-4,
        visualize=True,        # set True to log sample triplets occasionally
        visualize_every_n=1,   # steps between visual logs
    )
