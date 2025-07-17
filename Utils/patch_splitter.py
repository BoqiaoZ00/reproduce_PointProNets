import torch

def split_into_patches(vertices, faces, num_patches=3, patch_radius=10.0):
    """
    Splits the full mesh into local patches (vertices + faces), each covering a neighborhood.

    Args:
        vertices: (N, 3) torch.FloatTensor
        faces: (M, 3) torch.LongTensor
        num_patches: int, number of patches to sample
        patch_radius: float, radius for each patch

    Returns:
        patch_list: list of tuples (patch_vertices, patch_faces)
            - patch_vertices: (P, 3) torch.FloatTensor
            - patch_faces: (Q, 3) torch.LongTensor (local indices)
    """
    N = vertices.shape[0]
    patches = []

    # Randomly sample patch centers
    indices = torch.randperm(N)[:num_patches]
    centers = vertices[indices]

    for center in centers:
        # Step 1: Find vertices within radius
        distances = torch.norm(vertices - center, dim=1)
        mask = distances < patch_radius
        patch_vertex_indices = torch.nonzero(mask).squeeze(1)
        patch_vertices = vertices[patch_vertex_indices]

        # Step 2: Build a mapping from global vertex index → local index
        global_to_local = -torch.ones(N, dtype=torch.long)
        global_to_local[patch_vertex_indices] = torch.arange(patch_vertex_indices.size(0))

        # Step 3: Filter faces: keep only those with all vertices in the patch
        face_mask = mask[faces - 1].all(dim=1)
        patch_faces_global = faces[face_mask] + 1

        # Step 4: Remap face indices to local vertex indices
        patch_faces = global_to_local[patch_faces_global]

        # Skip if patch is too small or has no face
        if patch_vertices.size(0) >= 3 and patch_faces.size(0) > 0:
            patches.append((patch_vertices, patch_faces))

    return patches
