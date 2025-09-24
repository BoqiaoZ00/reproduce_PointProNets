import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


def project_points_to_heightmap_original(patch_list, normals, d_list=None, k=32, r=1.0, sigma=1.0):
    """
    Fully differentiable, paper-accurate projection to heightmap (Eq. 1–3).
    patch_list can be multiple patches, but must come from one item (has the same n)
    Args:
        patch_list: list of torch.FloatTensor, each of shape (Ni, 3) - All patches from one item
        normals: list or tensor [list shape (B, 3)] - one unit normal vector per patch
        d_list: List [list shape (B, 3)] or None - in-plane direction (optional, will be generated if None)
        k: int - output image resolution
        r: float - patch radius
        sigma: float - Gaussian std-dev for interpolation
    Returns:
        HN: (B, k, k) - heightmaps
    """
    B = len(patch_list)
    device = patch_list[0].device

    # Step 2: Interpolate onto discrete grid (Eq.3 setup)
    HN = torch.zeros((B, k, k), device=device)
    W = torch.zeros((B, k, k), device=device)

    # Create image grid coordinates (center positions)
    grid_coords = torch.stack(torch.meshgrid(
        torch.arange(k, device=device),
        torch.arange(k, device=device),
        indexing='ij'), dim=-1).float()  # (k, k, 2)

    grid_coords = grid_coords.view(-1, 2)  # (k², 2)
    grid_coords = grid_coords.unsqueeze(0).expand(B, -1, -1)  # make B copies (B, k², 2)

    for b, X in enumerate(patch_list):
        # Step 1: Construct frame (d, c, n)
        n = normals[b]
        if d_list is None or d_list[b] is None:
            up = torch.tensor([0, 0, 1.0], device=device)
            if torch.abs((n * up).sum()) > 0.9:
                up = torch.tensor([0, 1.0, 0], device=device)
            d = F.normalize(torch.cross(up, n, dim=0), dim=0)
        else:
            d = d_list[b].to(device)
        c = F.normalize(torch.cross(n, d, dim=0), dim=0)  # orthogonal vector

        Nb = X.size(0)

        # Project onto tangent plane at origin, push down by radius r (Eq. 1)
        dot_xn = (X * n).sum(dim=1, keepdim=True)  # (N, 1)
        P = X - (dot_xn + r) * n.unsqueeze(0)  # projected point on plane (N, 3)
        D = torch.norm(X - P, dim=1)  # (N, ) distance from original point

        # Convert projected coords to image (Eq. 2)
        pd = (P * d).sum(dim=1)  # (B, N)
        pc = (P * c).sum(dim=1)  # (B, N)
        scale = k / (2 * r)  # map [-r, r] → [0, k-1]
        i_x = ((pd + r) * scale).clamp(0, k - 1)
        i_y = ((pc + r) * scale).clamp(0, k - 1)

        # For each projected point, compute Gaussian-weighted sum to surrounding pixels
        for p_idx in range(Nb):
            pi = torch.stack([i_x[p_idx], i_y[p_idx]])  # (2,) The 2D coordinates of this points
            val = D[p_idx]  # scalar - The distance of this points

            # Compute distance to all pixel centers
            dists = torch.norm(grid_coords[b] - pi.unsqueeze(0), dim=1)  # (k²,)
            mask = dists < 3 * sigma  # restrict to nearby pixels (here we assume delta = 3*sigma)
            dists = dists[mask]
            grid_i = grid_coords[b][mask] # (M, 2) all M pixel centers that have Gaussian influence on the point pi

            weights = torch.exp(-(dists ** 2) / sigma ** 2)  # (M,)
            gx = grid_i[:, 0].long().clamp(0, k - 1)
            gy = grid_i[:, 1].long().clamp(0, k - 1)

            for idx in range(len(gx)):
                # print(weights[idx])
                # print(val)
                # print(HN[b, gx[idx], gy[idx]])
                HN[b, gx[idx], gy[idx]] += (weights[idx] * val)
                # print(HN[b, gx[idx], gy[idx]])
                W[b, gx[idx], gy[idx]] += weights[idx]
                # print(W[b, gx[idx], gy[idx]])

    # Safe division
    result = torch.where(W != 0, HN / W, torch.zeros_like(HN))
    return result


def project_points_to_heightmap_exact(
    patch_list, normals, d_list=None, k=32, r=1.0, sigma=1.0, eps=1e-8
):
    """
    Differentiable projection of 3D points to a kxk heightmap using Gaussian interpolation.

    Args:
        patch_list: list of (N, 3) tensors of 3D points (per batch item)
        normals:    list of (3,) tensors (surface normal per batch item)
        d_list:     list of (3,) tensors for in-plane x-axis d (optional). If None, computed from n.
        k:          image size
        r:          plane offset and scale half-range (maps [-r, r] → [0, k))
        sigma:      Gaussian sigma in pixel units
        eps:        small epsilon for safe division

    Returns:
        HN: (B, k, k) heightmaps (same device/dtype as inputs)
    """
    B = len(patch_list)
    device = patch_list[0].device
    dtype = patch_list[0].dtype

    # Grid of pixel centers in index space [0..k-1]
    i_coords, j_coords = torch.meshgrid(
        torch.arange(k, device=device, dtype=dtype),
        torch.arange(k, device=device, dtype=dtype),
        indexing='ij'
    )
    grid_coords = torch.stack([i_coords, j_coords], dim=-1)  # (k, k, 2)

    HN_list = []
    for b, X in enumerate(patch_list):
        n = F.normalize(normals[b], dim=0)

        # --- Build in-plane frame (d, c, n) ---
        if d_list is None or d_list[b] is None:
            # Continuous fallback: choose a helper vector and project to plane
            helper = torch.tensor([0.0, 0.0, 1.0], device=device, dtype=dtype)
            # If n ~ [0,0,1], this helper is collinear; blend with [0,1,0] smoothly
            alt = torch.tensor([0.0, 1.0, 0.0], device=device, dtype=dtype)
            print(n.shape)
            print(helper.shape)
            t = torch.clamp(n.abs().dot(helper), 0, 1)  # scalar in [0,1]
            base = F.normalize((1 - t) * helper + t * alt, dim=0)
            d_raw = base
        else:
            d_raw = d_list[b].to(device=device, dtype=dtype)

        # Orthogonalize d against n and normalize
        d = d_raw - (d_raw * n).sum() * n
        d = F.normalize(d, dim=0)

        # c completes the right-handed frame
        c = F.normalize(torch.linalg.cross(d, n), dim=0)

        # --- Project to plane offset by -r along n ---
        dot_xn = (X * n).sum(dim=1, keepdim=True)        # (N,1)
        P = X - (dot_xn + r) * n.unsqueeze(0)            # (N,3)
        D = torch.norm(X - P, dim=1)                     # (N,)

        # Debug Use only
        # D = X[:, 2]  # heights directly from z-coordinate

        # --- Map to image coordinates ---
        pd = (P * d).sum(dim=1)                          # (N,)
        pc = (P * c).sum(dim=1)                          # (N,)
        scale = k / (2.0 * r)                            # maps [-r,r] → [0,k)
        i_x = (pd + r) * scale
        i_y = (pc + r) * scale
        point_coords = torch.stack([i_x, i_y], dim=1)    # (N,2)

        # --- Gaussian interpolation in pixel-index space ---
        # distances from (i,j) pixel centers
        pc_exp = point_coords[:, None, None, :]          # (N,1,1,2)
        gc_exp = grid_coords[None, :, :, :]              # (1,k,k,2)
        dists_sq = ((pc_exp - gc_exp) ** 2).sum(dim=-1)  # (N,k,k)

        weights = torch.exp(-dists_sq / (sigma ** 2))    # (N,k,k)

        # Optional soft cutoff (keeps differentiability)
        thr2 = (3.0 * sigma) ** 2
        soft = torch.sigmoid(10.0 * (thr2 - dists_sq))
        weights = weights * soft

        # Weighted average of distances D
        num = (weights * D[:, None, None]).sum(dim=0)    # (k,k)
        den = weights.sum(dim=0)                         # (k,k)
        HN_b = num / (den + eps)

        HN_list.append(HN_b)

    HN = torch.stack(HN_list, dim=0)                     # (B,k,k)
    return HN


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


def test_project_points_to_heightmap_exact_basic():
    """Test basic functionality with simple input - verify equations precisely."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create a simple patch: a single point at (0, 0, 1)
    patch = torch.tensor([[0.0, 0.0, 1.0]], device=device)
    normal = torch.tensor([0.0, 0.0, 1.0], device=device)

    # Option 1: Let the function generate d automatically and verify
    heightmap_auto = project_points_to_heightmap_exact([patch], [normal], k=32, r=1.0, sigma=0.1)

    # With n = (0,0,1), the auto-generated d should be (1,0,0) and c = (0,1,0)
    # So the calculations from before should hold:
    # p = [0,0,-1], D = 2, i_x = 16, i_y = 16

    center_x, center_y = 16, 16
    center_value = heightmap_auto[0, center_x, center_y]

    # The value should be approximately 2 (the distance)
    assert torch.isclose(center_value, torch.tensor(2.0, device=device), atol=0.1), \
        f"Expected value ≈ 2.0, got {center_value.item()}"

    # Option 2: Pass a specific d_list for deterministic testing
    d_list = [torch.tensor([1.0, 0.0, 0.0], device=device)]
    heightmap_fixed = project_points_to_heightmap_exact([patch], [normal], d_list=d_list, k=32, r=1.0, sigma=0.1)

    # Should give the same result
    center_value_fixed = heightmap_fixed[0, center_x, center_y]
    assert torch.isclose(center_value_fixed, torch.tensor(2.0, device=device), atol=0.1)
    assert torch.allclose(heightmap_auto, heightmap_fixed, atol=1e-6)

def test_project_points_to_heightmap_exact_with_custom_d():
    """Test with custom d_list to verify coordinate mapping."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create a point at (1, 0, 0) - on the x-axis
    patch = torch.tensor([[1.0, 0.0, 0.0]], device=device)
    normal = torch.tensor([1.0, 0.0, 0.0], device=device)  # Normal along x-axis

    # Test with different d directions to see how it affects the projection
    d_test_cases = [
        # d, expected x-coordinate behavior
        (torch.tensor([0.0, 1.0, 0.0], device=device), "should project to center in x"),
        (torch.tensor([0.0, 0.0, 1.0], device=device), "should project to edge in x"),
    ]

    for d, description in d_test_cases:
        d_list = [d]
        heightmap = project_points_to_heightmap_exact([patch], [normal], d_list=d_list, k=32, r=1.0, sigma=0.5)

        # x (1, 0, 0) with n (1, 0, 0)
        # Projection: p = [1,0,0] - ([1,0,0]·[1,0,0] + 1)[1,0,0] = [1,0,0] - (1+1)[1,0,0] = [-1,0,0]
        # Distance: D = ||[1,0,0] - [-1,0,0]|| = 2

        # Image coordinates depend on d and c = cross(n, d)
        if torch.allclose(d, torch.tensor([0.0, 1.0, 0.0], device=device)):
            # d = (0,1,0), c = cross((1,0,0), (0,1,0)) = (0,0,1)
            # i_x = 16 * (p·d + 1) = 16 * ([-1,0,0]·[0,1,0] + 1) = 16 * (0 + 1) = 16
            # i_y = 16 * (p·c + 1) = 16 * ([-1,0,0]·[0,0,1] + 1) = 16 * (0 + 1) = 16
            expected_x, expected_y = 16, 16
        else:  # d = (0,0,1)
            # d = (0,0,1), c = cross((1,0,0), (0,0,1)) = (0,-1,0)
            # i_x = 16 * (p·d + 1) = 16 * ([-1,0,0]·[0,0,1] + 1) = 16 * (0 + 1) = 16
            # i_y = 16 * (p·c + 1) = 16 * ([-1,0,0]·[0,-1,0] + 1) = 16 * (0 + 1) = 16
            expected_x, expected_y = 16, 16

        # Check that the maximum value is near the expected coordinates
        max_val, max_idx = heightmap[0].max(), heightmap[0].argmax()
        max_x, max_y = max_idx // 32, max_idx % 32

        assert abs(max_x - expected_x) <= 1 and abs(max_y - expected_y) <= 1, \
            f"With d={d.cpu().numpy()}, expected max near ({expected_x},{expected_y}), got ({max_x},{max_y})"

        # Value should be approximately 2
        assert torch.isclose(max_val, torch.tensor(2.0, device=device), atol=0.2), \
            f"Expected value ≈ 2.0, got {max_val.item()}"

def test_project_points_to_heightmap_exact_two_points_same_location():
    """Test two points at the same 3D location but different distances."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Two points at the same (x,y) but different z (different distances)
    points = torch.tensor([
        [0.0, 0.0, 1.0],  # Distance = 2.0 (from projection math)
        [0.0, 0.0, 2.0],  # Distance = 3.0 (from projection math)
    ], device=device)

    normal = torch.tensor([0.0, 0.0, 1.0], device=device)
    d_list = [torch.tensor([1.0, 0.0, 0.0], device=device)]  # Fixed for predictability

    heightmap = project_points_to_heightmap_exact([points], [normal], d_list=d_list, k=32, r=1.0, sigma=0.5)

    # Hand computation:
    # For both points: p = [0,0,z] - (z + 1)[0,0,1] = [0,0,-1]
    # Distance for first point: ||[0,0,1] - [0,0,-1]|| = 2.0
    # Distance for second point: ||[0,0,2] - [0,0,-1]|| = 3.0
    # Both project to same image coordinates: (16, 16)

    # Gaussian weights for both points at (16,16):
    # g(16,16,16,16) = exp(-0/0.25) = 1.0 for both
    # So weighted average = (1.0*2.0 + 1.0*3.0) / (1.0 + 1.0) = 2.5

    center_value = heightmap[0, 16, 16]
    assert torch.isclose(center_value, torch.tensor(2.5, device=device), atol=1e-6), \
        f"Expected weighted average of 2.5, got {center_value.item()}"

def test_project_points_to_heightmap_exact_two_points_different_locations():
    """Test two points at different locations with hand-computable Gaussian interpolation."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Two points that will project to specific pixel coordinates
    points = torch.tensor([
        [0.0, 0.0, 1.0],  # Projects to (16, 16), distance = 2.0
        [0.5, 0.0, 1.0],  # Projects to (24, 16), distance = 2.0
    ], device=device)

    normal = torch.tensor([0.0, 0.0, 1.0], device=device)
    d_list = [torch.tensor([1.0, 0.0, 0.0], device=device)]

    heightmap = project_points_to_heightmap_exact([points], [normal], d_list=d_list, k=32, r=1.0, sigma=2.0)

    # Hand computation:
    # Point 1: p = [0,0,-1], D = 2.0, coords = (16, 16)
    # Point 2: p = [0.5,0,-1], D = 2.0, coords = (24, 16)

    # Check pixel (20, 16) - midpoint between the two points
    # Distance from (20,16) to (16,16): 4.0 pixels
    # Distance from (20,16) to (24,16): 4.0 pixels
    # Weight for point 1: exp(-16/1) = exp(-16) ≈ 1.125e-7
    # Weight for point 2: exp(-16/1) = exp(-16) ≈ 1.125e-7
    # Value at (20,16) ≈ (1.125e-7*2.0 + 1.125e-7*2.0) / (1.125e-7 + 1.125e-7) = 2.0

    midpoint_value = heightmap[0, 20, 16]
    assert torch.isclose(midpoint_value, torch.tensor(2.0, device=device), atol=1e-6), \
        f"Expected 2.0 at midpoint, got {midpoint_value.item()}"

    # Check pixel (16, 16) - exact location of point 1
    # Weight for point 1: exp(0) = 1.0
    # Weight for point 2: exp(-64/1) = exp(-64) ≈ 1.603e-28 (negligible)
    # Value at (16,16) ≈ (1.0*2.0 + negligible*2.0) / (1.0 + negligible) ≈ 2.0

    point1_value = heightmap[0, 16, 16]
    assert torch.isclose(point1_value, torch.tensor(2.0, device=device), atol=1e-6), \
        f"Expected 2.0 at point 1 location, got {point1_value.item()}"

def test_project_points_to_heightmap_exact_three_points_triangle():
    """Test three points forming a triangle with hand-computable interpolation."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Three points forming an equilateral triangle in the projection plane
    points = torch.tensor([
        [0.0, 0.0, 1.0],  # Center, distance = 2.0
        [0.5, 0.0, 1.0],  # Right, distance = 2.0
        [0.25, 0.433, 1.0],  # Top-right, distance = 2.0
    ], device=device)

    normal = torch.tensor([0.0, 0.0, 1.0], device=device)
    d_list = [torch.tensor([1.0, 0.0, 0.0], device=device)]

    heightmap = project_points_to_heightmap_exact([points], [normal], d_list=d_list, k=32, r=1.0, sigma=2.0)

    # Hand computation for centroid of triangle (approximately 18, 18)
    # Point 1: (16, 16), distance = 4.472 pixels to (18,18)
    # Point 2: (24, 16), distance = 6.325 pixels to (18,18)
    # Point 3: (20, 24), distance = 6.325 pixels to (18,18)

    # Weights: exp(-distance²/4)
    # w1 = exp(-20/4) = exp(-5) ≈ 0.0067
    # w2 = exp(-40/4) = exp(-10) ≈ 4.54e-5
    # w3 = exp(-40/4) = exp(-10) ≈ 4.54e-5

    # Weighted average = (0.0067*2.0 + 4.54e-5*2.0 + 4.54e-5*2.0) / (0.0067 + 4.54e-5 + 4.54e-5)
    # ≈ (0.0134 + 9.08e-5 + 9.08e-5) / 0.00679 ≈ 0.01358 / 0.00679 ≈ 2.0

    centroid_value = heightmap[0, 18, 18]
    assert torch.isclose(centroid_value, torch.tensor(2.0, device=device), atol=0.1), \
        f"Expected ≈2.0 at centroid, got {centroid_value.item()}"

def test_project_points_to_heightmap_exact_mixed_distances():
    """Test points with different distances at the same location."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Points at same (x,y) but different distances
    points = torch.tensor([
        [0.0, 0.0, 1.0],  # Distance = 2.0
        [0.0, 0.0, 2.0],  # Distance = 3.0
        [0.0, 0.0, 0.5],  # Distance = 1.5
    ], device=device)

    normal = torch.tensor([0.0, 0.0, 1.0], device=device)
    d_list = [torch.tensor([1.0, 0.0, 0.0], device=device)]

    # Test with different sigma values - all should give same result since points
    # map to exactly the same pixel coordinates
    for sigma in [0.1, 1.0, 10.0]:
        heightmap = project_points_to_heightmap_exact([points], [normal], d_list=d_list,
                                                      k=32, r=1.0, sigma=sigma)

        center_value = heightmap[0, 16, 16]

        # All points map to exactly (16,16), so Gaussian weights are all exp(0) = 1.0
        # regardless of sigma. The result should be the average of distances.
        expected = (2.0 + 3.0 + 1.5) / 3.0  # = 2.1667

        assert torch.isclose(center_value, torch.tensor(expected, device=device), atol=1e-6), \
            f"With sigma={sigma}, expected {expected}, got {center_value.item()}"

def test_project_points_to_heightmap_exact_different_normal_orientation():
    """Test with different normal orientation and hand-computed results."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Point at (1, 0, 0) with normal along x-axis
    point = torch.tensor([[1.0, 0.0, 0.0]], device=device)
    normal = torch.tensor([1.0, 0.0, 0.0], device=device)
    d_list = [torch.tensor([0.0, 1.0, 0.0], device=device)]  # d along y-axis

    heightmap = project_points_to_heightmap_exact([point], [normal], d_list=d_list, k=32, r=1.0, sigma=0.1)

    # Hand computation:
    # Projection: p = [1,0,0] - ([1,0,0]·[1,0,0] + 1)[1,0,0] = [1,0,0] - (1+1)[1,0,0] = [-1,0,0]
    # Distance: D = ||[1,0,0] - [-1,0,0]|| = 2.0

    # c = cross(n, d) = cross([1,0,0], [0,1,0]) = [0,0,-1]
    # Image coordinates:
    # i_x = 16 * (p·d + 1) = 16 * ([-1,0,0]·[0,1,0] + 1) = 16 * (0 + 1) = 16
    # i_y = 16 * (p·c + 1) = 16 * ([-1,0,0]·[0,0,-1] + 1) = 16 * (0 + 1) = 16

    center_value = heightmap[0, 16, 16]
    assert torch.isclose(center_value, torch.tensor(2.0, device=device), atol=0.1), \
        f"Expected 2.0 at center, got {center_value.item()}"

def test_project_points_to_heightmap_exact_edge_case_r_adjustment():
    """Test that the r offset works correctly for edge cases."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Point exactly at the origin
    point = torch.tensor([[0.0, 0.0, 0.0]], device=device)
    normal = torch.tensor([0.0, 0.0, 1.0], device=device)
    d_list = [torch.tensor([1.0, 0.0, 0.0], device=device)]

    heightmap = project_points_to_heightmap_exact([point], [normal], d_list=d_list, k=32, r=1.0, sigma=0.1)

    # Hand computation:
    # Projection: p = [0,0,0] - ([0,0,0]·[0,0,1] + 1)[0,0,1] = [0,0,0] - (0+1)[0,0,1] = [0,0,-1]
    # Distance: D = ||[0,0,0] - [0,0,-1]|| = 1.0
    # Image coordinates: (16, 16)

    center_value = heightmap[0, 16, 16]
    assert torch.isclose(center_value, torch.tensor(1.0, device=device), atol=0.1), \
        f"Expected 1.0 at center, got {center_value.item()}"

# def test_project_points_to_heightmap_exact_boundary_conditions():
#     """Test points at the boundary of the projection area."""
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#
#     # Points at the extreme boundaries of the [-r, r] range
#     test_cases = [
#         # (point, expected behavior)
#         ([-1.0, 0.0, 0.0], "left edge"),  # Should map to x=0
#         ([1.0, 0.0, 0.0], "right edge"),  # Should map to x=31
#         ([0.0, -1.0, 0.0], "bottom edge"),  # Should map to y=0
#         ([0.0, 1.0, 0.0], "top edge"),  # Should map to y=31
#         ([-1.0, -1.0, 0.0], "bottom-left"),  # Should map to (0,0)
#         ([1.0, 1.0, 0.0], "top-right"),  # Should map to (31,31)
#     ]
#
#     normal = torch.tensor([0.0, 0.0, 1.0], device=device)
#     d_list = [torch.tensor([1.0, 0.0, 0.0], device=device)]
#
#     for point_coords, description in test_cases:
#         point = torch.tensor([point_coords], device=device)
#         heightmap = project_points_to_heightmap_exact([point], [normal], d_list=d_list,
#                                                           k=32, r=1.0, sigma=0.1)
#
#         # Hand computation for expected image coordinates
#         # p = x - (x·n + r)n = (x,y,0) - (0 + 1)(0,0,1) = (x,y,-1)
#         # i_x = 16 * (p·d + 1) = 16 * (x + 1)
#         # i_y = 16 * (p·c + 1) = 16 * (y + 1)
#
#         x, y, _ = point_coords
#         expected_x = int(round(16 * (x + 1)))
#         expected_y = int(round(16 * (y + 1)))
#
#         # Clamp to valid coordinates
#         expected_x = min(max(expected_x, 0), 31)
#         expected_y = min(max(expected_y, 0), 31)
#
#         # The point should contribute significantly to the expected pixel
#         pixel_value = heightmap[0, expected_x, expected_y]
#         assert pixel_value > 0.1, \
#             f"{description}: Expected significant value at ({expected_x},{expected_y}), got {pixel_value.item()}"

def test_project_points_to_heightmap_exact_origin_behavior():
    """Test the mathematical behavior at the origin with different r values."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Point at origin
    point = torch.tensor([[0.0, 0.0, 0.0]], device=device)
    normal = torch.tensor([0.0, 0.0, 1.0], device=device)
    d_list = [torch.tensor([1.0, 0.0, 0.0], device=device)]

    for r in [0.5, 1.0, 2.0]:
        heightmap = project_points_to_heightmap_exact([point], [normal], d_list=d_list,
                                                          k=32, r=r, sigma=0.1)

        # Hand computation:
        # p = (0,0,0) - (0 + r)(0,0,1) = (0,0,-r)
        # D = ||(0,0,0) - (0,0,-r)|| = r
        # i_x = (k/(2r)) * (0 + r) = (32/(2r)) * r = 16
        # i_y = (k/(2r)) * (0 + r) = 16

        center_value = heightmap[0, 16, 16]
        assert torch.isclose(center_value, torch.tensor(r, device=device), atol=1e-6), \
            f"With r={r}, expected {r} at center, got {center_value.item()}"

def test_project_points_to_heightmap_exact_negative_r_handling():
    """Test that the function handles the -r offset correctly."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Test various points to verify the -r offset works as expected
    test_points = [
        [0.0, 0.0, 0.0],  # At origin
        [0.0, 0.0, 0.5],  # Above origin
        [0.0, 0.0, -0.5],  # Below origin
        [0.5, 0.0, 0.0],  # Right of origin
    ]

    normal = torch.tensor([0.0, 0.0, 1.0], device=device)
    d_list = [torch.tensor([1.0, 0.0, 0.0], device=device)]

    for point_coords in test_points:
        point = torch.tensor([point_coords], device=device)
        heightmap = project_points_to_heightmap_exact([point], [normal], d_list=d_list,
                                                          k=32, r=1.0, sigma=0.1)

        x, y, z = point_coords

        # Hand computation with -r offset:
        # p = (x,y,z) - ((x,y,z)·(0,0,1) + 1)(0,0,1) = (x,y,z) - (z + 1)(0,0,1) = (x,y,-1)
        # D = ||(x,y,z) - (x,y,-1)|| = |z + 1|
        # i_x = 16 * (x + 1)
        # i_y = 16 * (y + 1)

        expected_D = abs(z + 1)
        expected_x = int(round(16 * (x + 1)))
        expected_y = int(round(16 * (y + 1)))

        # Clamp to valid coordinates
        expected_x = min(max(expected_x, 0), 31)
        expected_y = min(max(expected_y, 0), 31)

        # Check the value at the expected pixel
        pixel_value = heightmap[0, expected_x, expected_y]
        assert torch.isclose(pixel_value, torch.tensor(expected_D, device=device), atol=0.1), \
            f"Point {point_coords}: expected {expected_D} at ({expected_x},{expected_y}), got {pixel_value.item()}"


# Run the value-based tests
if __name__ == "__main__":
    test_project_points_to_heightmap_exact_basic()
    test_project_points_to_heightmap_exact_with_custom_d()

    test_project_points_to_heightmap_exact_two_points_same_location()
    test_project_points_to_heightmap_exact_two_points_different_locations()
    test_project_points_to_heightmap_exact_three_points_triangle()
    test_project_points_to_heightmap_exact_mixed_distances()
    test_project_points_to_heightmap_exact_different_normal_orientation()
    test_project_points_to_heightmap_exact_edge_case_r_adjustment()

    # test_project_points_to_heightmap_exact_boundary_conditions()
    test_project_points_to_heightmap_exact_origin_behavior()
    test_project_points_to_heightmap_exact_negative_r_handling()

    print("All value-based tests passed!")