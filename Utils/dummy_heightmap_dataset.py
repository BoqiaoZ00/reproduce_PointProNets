import math
import random
from typing import Tuple

import torch
from torch.utils.data import Dataset


class DummyHeightmapDataset(Dataset):
    """
    Synthetic heightmap dataset that generates simple clean patterns and adds noise.

    Each item returns a tuple: (heightmap_gt, heightmap_noisy, normal_support_placeholder)

    - heightmap_gt: torch.FloatTensor of shape (1, k, k)
    - heightmap_noisy: torch.FloatTensor of shape (1, k, k)
    - normal_support_placeholder: torch.FloatTensor of shape (3,) (unused by the denoiser)
    """

    def __init__(
        self,
        num_samples: int = 2000,
        k: int = 64,
        noise_std: float = 0.05,
        seed: int = 0,
    ) -> None:
        super().__init__()
        self.num_samples = int(num_samples)
        self.k = int(k)
        self.noise_std = float(noise_std)
        self.random = random.Random(seed)

        # Precompute coordinate grid in [-1, 1]
        lin = torch.linspace(-1.0, 1.0, self.k)
        self.yy, self.xx = torch.meshgrid(lin, lin, indexing="ij")  # (k, k)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int):
        clean = self._generate_clean_heightmap()
        noisy = self._add_noise(clean)

        # The training loop expects a 3-tuple
        normal_support_placeholder = torch.zeros(3, dtype=torch.float32)
        return clean.unsqueeze(0), noisy.unsqueeze(0), normal_support_placeholder

    def _generate_clean_heightmap(self) -> torch.Tensor:
        """
        Compose a clean heightmap from a few simple procedural primitives:
        - tilted plane
        - sinusoidal ripples
        - gaussian bump(s)
        - circular/radial wave
        """
        k = self.k
        x = self.xx
        y = self.yy

        # Start with a gentle tilted plane
        ax = self._rand_uniform(-0.4, 0.4)
        ay = self._rand_uniform(-0.4, 0.4)
        plane = ax * x + ay * y

        # Add sinusoidal components
        sx = self._rand_uniform(1.0, 4.0)
        sy = self._rand_uniform(1.0, 4.0)
        phase_x = self._rand_uniform(0.0, 2.0 * math.pi)
        phase_y = self._rand_uniform(0.0, 2.0 * math.pi)
        amp_sin = self._rand_uniform(0.1, 0.4)
        sinxy = amp_sin * (
            torch.sin(sx * math.pi * x + phase_x) + torch.sin(sy * math.pi * y + phase_y)
        )

        # Add a gaussian bump (or two) at random positions
        num_bumps = 1 if self.random.random() < 0.6 else 2
        bumps = torch.zeros_like(x)
        for _ in range(num_bumps):
            cx = self._rand_uniform(-0.5, 0.5)
            cy = self._rand_uniform(-0.5, 0.5)
            sigma = self._rand_uniform(0.1, 0.4)
            amp = self._rand_uniform(0.1, 0.5)
            r2 = (x - cx) ** 2 + (y - cy) ** 2
            bumps = bumps + amp * torch.exp(-r2 / (2.0 * sigma * sigma))

        # Optional circular wave
        if self.random.random() < 0.5:
            freq = self._rand_uniform(2.0, 5.0)
            amp_ring = self._rand_uniform(0.05, 0.2)
            r = torch.sqrt(x ** 2 + y ** 2)
            ring = amp_ring * torch.sin(freq * math.pi * r)
        else:
            ring = torch.zeros_like(x)

        clean = plane + sinxy + bumps + ring

        # Normalize to roughly [-1, 1] for stability
        clean = clean - clean.mean()
        std = clean.std()
        if std > 1e-6:
            clean = clean / (2.5 * std)
        clean = clean.clamp(-1.0, 1.0)
        return clean.to(torch.float32)

    def _add_noise(self, clean: torch.Tensor) -> torch.Tensor:
        # Gaussian noise
        noise = torch.randn_like(clean) * self.noise_std

        # Optionally add low-probability salt & pepper like outliers
        if self.random.random() < 0.3:
            prob_spike = self._rand_uniform(0.001, 0.01)
            mask_hi = torch.rand_like(clean) < prob_spike
            mask_lo = torch.rand_like(clean) < prob_spike
            spike_val_hi = self._rand_uniform(0.6, 1.2)
            spike_val_lo = -self._rand_uniform(0.6, 1.2)
            noise = noise + spike_val_hi * mask_hi.to(clean.dtype) + spike_val_lo * mask_lo.to(clean.dtype)

        noisy = (clean + noise).clamp(-1.5, 1.5)
        return noisy

    def _rand_uniform(self, lo: float, hi: float) -> float:
        return self.random.uniform(lo, hi)

