import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


def get_normal(X):
    """
    X: (B, N, 3)
    Returns:
        n: (B, 3) - one unit normal vector per patch
    """
    B, N, _ = X.shape
    n = torch.randn(1, 3, device=X.device)
    n = F.normalize(n, dim=1)  # Make it unit-length
    return n


def project_points_to_heightmap_exact(X, n, d=None, k=32, r=1.0, sigma=1.0):
    """
    Fully differentiable, paper-accurate projection to heightmap (Eq. 1–3).
    Args:
        X: (B, N, 3) - input points
        n: (1, 3) - !unit! normal vectors
        d: (B, 3) or None - in-plane direction (optional, will be generated if None)
        k: int - output image resolution
        r: float - patch radius
        sigma: float - Gaussian std-dev for interpolation
    Returns:
        HN: (B, 1, k, k) - heightmaps
    """
    B, N, _ = X.shape
    device = X.device

    # Step 1: Construct frame (d, c, n)
    if d is None:
        up = torch.tensor([0, 0, 1.0], device=device).expand(B, 3)
        parallel = (torch.abs((n * up).sum(dim=1)) > 0.9)
        up[parallel] = torch.tensor([0, 1.0, 0], device=device)
        d = F.normalize(torch.cross(up, n, dim=1), dim=1)
    c = F.normalize(torch.cross(n, d, dim=1), dim=1)  # orthogonal vector

    # Step 2: Project points onto plane
    dot_xn = (X * n.unsqueeze(1)).sum(dim=2, keepdim=True)  # (B, N, 1)
    P = X - (dot_xn + r) * n.unsqueeze(1)  # projected point on plane (B, N, 3)
    D = torch.norm(X - P, dim=2)  # (B, N) distance from original point

    # Step 3: Convert projected coords to image (Eq. 2)
    pd = (P * d.unsqueeze(1)).sum(dim=2)  # (B, N)
    pc = (P * c.unsqueeze(1)).sum(dim=2)  # (B, N)
    i_x = ((pd + r) * (k / (2 * r))).clamp(0, k - 1)
    i_y = ((pc + r) * (k / (2 * r))).clamp(0, k - 1)

    # Step 4: Interpolate onto discrete grid using Equation (3)
    HN = torch.zeros((B, k, k), device=device)
    W = torch.zeros((B, k, k), device=device)

    # Create image grid coordinates (center positions)
    grid_coords = torch.stack(torch.meshgrid(
        torch.arange(k, device=device),
        torch.arange(k, device=device),
        indexing='ij'), dim=-1).float()  # (k, k, 2)

    grid_coords = grid_coords.view(-1, 2)  # (k², 2)
    grid_coords = grid_coords.unsqueeze(0).expand(B, -1, -1)  # make B copies (B, k², 2)

    for b in range(B):
        # Get projected points for this batch
        points_i = torch.stack([i_x, i_y], dim=1)  # (N, 2)
        dist_vals = D  # (N,)

        # For each projected point, compute Gaussian-weighted sum to surrounding pixels
        for p_idx in range(points_i.size(0)):
            pi = points_i[p_idx]  # (2,)
            val = dist_vals[p_idx]  # scalar

            # Compute distance to all pixel centers
            dists = torch.norm(grid_coords[b] - pi.unsqueeze(0), dim=1)  # (k²,)
            mask = dists < 3 * sigma  # restrict to nearby pixels (here we assume delta = 3*sigma)
            dists = dists[mask]
            grid_i = grid_coords[b][mask]  # (M, 2) all M pixel centers that have Gaussian influence on the point pi

            weights = torch.exp(-(dists ** 2) / sigma ** 2)  # (M,)
            gx = grid_i[:, 0].long().clamp(0, k - 1)
            gy = grid_i[:, 1].long().clamp(0, k - 1)

            for idx in range(len(gx)):
                HN[b, gx[idx], gy[idx]] += weights[idx] * val
                W[b, gx[idx], gy[idx]] += weights[idx]

    # Safe division
    result = torch.where(W != 0, HN / W, torch.zeros_like(HN))
    return result


class FrameEstimatorNet(nn.Module):
    def __init__(self):
        super(FrameEstimatorNet, self).__init__()

        self.conv1 = nn.Conv1d(3, 64, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv1d(64, 128, 1)
        self.bn2 = nn.BatchNorm1d(128)
        self.relu2 = nn.ReLU(inplace=True)

        self.conv3 = nn.Conv1d(128, 256, 1)
        self.bn3 = nn.BatchNorm1d(256)
        self.relu3 = nn.ReLU(inplace=True)

        self.conv4 = nn.Conv1d(256, 512, 1)
        self.bn4 = nn.BatchNorm1d(512)
        self.relu4 = nn.ReLU(inplace=True)

        self.conv5 = nn.Conv1d(512, 1024, 1)
        self.bn5 = nn.BatchNorm1d(1024)
        self.relu5 = nn.ReLU(inplace=True)

        self.fc1 = nn.Linear(1024, 512)
        self.relu6 = nn.ReLU(inplace=True)

        self.fc2 = nn.Linear(512, 256)
        self.relu7 = nn.ReLU(inplace=True)

        self.fc3 = nn.Linear(256, 3)

    def forward(self, x, n_gt=None, training=False):
        x = x - x.mean(dim=1, keepdim=True)
        x = x.transpose(1, 2) # should pay extra attention here about the order of (B_patches, N_numpoints, 3_xyz)

        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.relu4(self.bn4(self.conv4(x)))
        x = self.relu5(self.bn5(self.conv5(x)))

        x = torch.max(x, dim=2)[0] # should pay extra attention here about the dimension to max pooling

        x = self.relu6(self.fc1(x))
        x = self.relu7(self.fc2(x))
        n_pred = self.fc3(x)
        n_pred = F.normalize(n_pred, p=2, dim=1)

        if training and n_gt is not None:
            dot = (n_pred * n_gt).sum(dim=1, keepdim=True)
            n_pred = n_pred * torch.sign(dot)

        return n_pred
