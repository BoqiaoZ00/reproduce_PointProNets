import torch

def normalize_mesh(vertices, method="unit_cube"):
    """
    Normalize mesh vertices but keep density

    Args:
        vertices: (N, 3) torch.FloatTensor
        method:   str, normalization method ('unit_cube' or 'unit_sphere')

    Returns:
        norm_vertices: (N, 3) torch.FloatTensor (normalized)
        faces: unchanged
        norm_params: dict with scale and translation (for inverse transform)
    """
    verts = vertices.clone()

    if method == "unit_cube":
        # Center at origin
        centroid = verts.mean(dim=0)
        verts = verts - centroid

        # Scale to fit inside [-1, 1]
        max_range = verts.abs().max()
        verts = verts / max_range

        norm_params = {"centroid": centroid, "scale": max_range}

    elif method == "unit_sphere":
        centroid = verts.mean(dim=0)
        verts = verts - centroid

        scale = torch.norm(verts, dim=1).max()
        verts = verts / scale

        norm_params = {"centroid": centroid, "scale": scale}

    else:
        raise ValueError(f"Unknown normalization method: {method}")

    return verts, norm_params
