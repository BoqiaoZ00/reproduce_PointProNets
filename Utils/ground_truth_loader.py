import torch
import HeightmapGenerator as HGN
from torch.nn.functional import normalize
import numpy as np
from collections import defaultdict


def compute_gt_normals(vertices, faces):
    """
    Compute per-vertex normals from mesh faces by averaging adjacent face normals.

    Args:
        vertices: (N, 3) torch tensor of vertex positions
        faces: (M, 3) torch tensor of triangle indices (1-based)

    Returns:
        normals: (N, 3) torch tensor of unit normals
    """
    device = vertices.device
    N = vertices.size(0)
    normals = torch.zeros((N, 3), dtype=torch.float32, device=device)

    for f in faces:
        v0, v1, v2 = vertices[f[0] - 1], vertices[f[1] - 1], vertices[f[2] - 1]
        e1 = v1 - v0
        e2 = v2 - v0
        face_normal =  torch.linalg.cross(e1, e2)
        face_normal = face_normal / torch.norm(face_normal)

        for i in f:
            normals[i-1] += face_normal

    normals = torch.nn.functional.normalize(normals, dim=1)
    return normals


def compute_gt_heightmap(vertices, faces, n = None, k=32, r=1.0, sigma=1.0):
    """
    Generate ground truth heightmap from a clean patch.

    Args:
        vertices: (N, 3) torch tensor of point positions
        faces: (M, 3) torch tensor of triangle indices (1-based indices of vertices)
        n: torch.Size([3])  The pre-computed global ground truth normal (ngt) for this patch (also for the whole item)
        k: grid size (heightmap will be k x k)
        r: radius offset (used to shift the projection plane)
        sigma: std-dev of Gaussian kernel for interpolation

    Returns:
        H_GT: (k, k) torch tensor
    """
    B = 1 # only for one patch
    device = vertices.device
    N = vertices.shape[0]

    # 1. If no pre-computed global n_GT for the item, compute n_GT for this patch
    if n is None:
        normals = compute_gt_normals(vertices, faces)
        n = torch.mean(normals, dim=0)

    # Below is essentially just HGN projector with dimensional simplifications
    # 2. Choose random orthogonal d to n
    up = torch.tensor([0., 0., 1.], device=device)
    if torch.abs(torch.dot(n, up)) > 0.9:
        up = torch.tensor([0., 1., 0.], device=device)
    d = torch.nn.functional.normalize(torch.linalg.cross(up, n), dim=0)
    c = torch.linalg.cross(n, d)

    # 3. Project points to plane
    xn = (vertices @ n).unsqueeze(1)  # (N,1)
    p = vertices - (xn + r) * n.unsqueeze(0)  # (N,3)

    # 4. Image coordinates
    pd = (p @ d) + r  # (N,)
    pc = (p @ c) + r
    i_x = (pd * k / (2 * r)).long().clamp(0, k - 1)
    i_y = (pc * k / (2 * r)).long().clamp(0, k - 1)

    # 5. Heights (distance from original point to projection)
    D = torch.norm(vertices - p, dim=1)  # (N,)

    # 6. Interpolate using Gaussian
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
    return result[0], n # result[0] is (k, k) because it's for one patch, n is torch.Size([3])
