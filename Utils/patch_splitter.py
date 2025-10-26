import torch

def split_into_patches(vertices, faces, num_patches=3, patch_radius=0.05):
    """
    Splits the full mesh into local patches (vertices + faces), each covering a neighborhood.

    Args:
        vertices: (N, 3) torch.FloatTensor
        faces: (M, 3) torch.LongTensor (1-based indices!)
        num_patches: int, number of patches to sample
        patch_radius: float, radius for each patch

    Returns:
        patch_list: list of tuples (patch_vertices, patch_faces)
            - patch_vertices: (P, 3) torch.FloatTensor
            - patch_faces: (Q, 3) torch.LongTensor (1-based local indices)
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
        patch_vertex_indices = torch.nonzero(mask).squeeze(1)  # 0-based
        patch_vertices = vertices[patch_vertex_indices]

        if patch_vertex_indices.numel() == 0:
            continue  # skip empty patch

        # Step 2: Filter faces where all 3 vertices are in the patch
        # faces: 1-based, so subtract 1 for indexing
        face_mask = mask[faces - 1].all(dim=1)
        selected_faces = faces[face_mask]  # still 1-based global indices

        # Step 3: Map global 1-based vertex index to local 1-based index
        global_patch_ids = patch_vertex_indices + 1  # convert to 1-based
        global_to_local = {int(g.item()): i + 1 for i, g in enumerate(global_patch_ids)}  # 1-based map

        remapped_faces = []
        for face in selected_faces:
            f0, f1, f2 = face.tolist()
            if f0 in global_to_local and f1 in global_to_local and f2 in global_to_local:
                remapped_faces.append([
                    global_to_local[f0],
                    global_to_local[f1],
                    global_to_local[f2]
                ])

        if len(remapped_faces) == 0:
            continue

        patch_faces = torch.tensor(remapped_faces, dtype=torch.long)  # still 1-based
        patches.append((patch_vertices, patch_faces))

    return patches


def split_thimble_into_patches(vertices, normals, num_patches, patch_radius):
    """
    Splits the full point cloud (vertices + normals) into local patches.

    Args:
        vertices: (N, 3) torch.FloatTensor of vertex positions
        normals: (N, 3) torch.FloatTensor of per-vertex normals
        num_patches: int, number of patches to sample
        patch_radius: float, radius for each patch

    Returns:
        patch_list: list of tuples (patch_vertices, patch_normals)
            - patch_vertices: (P, 3) torch.FloatTensor
            - patch_normals: (P, 3) torch.FloatTensor
    """
    N = vertices.shape[0]
    if N == 0:
        return []

    patches = []

    # Randomly sample patch centers
    # Use randperm on the device of the tensors to avoid CPU-GPU sync
    indices = torch.randperm(N, device=vertices.device)[:num_patches]
    centers = vertices[indices]

    for center in centers:
        # Step 1: Find vertices within radius
        # (N, 3) vertices - (1, 3) center -> (N, 3) distances
        distances = torch.norm(vertices - center, dim=1)
        mask = distances < patch_radius

        # Get the 0-based indices of all points in the patch
        patch_vertex_indices = torch.nonzero(mask).squeeze(1)

        if patch_vertex_indices.numel() == 0:
            continue  # skip empty patch

        # Step 2: Grab the vertices and normals using the indices
        patch_vertices = vertices[patch_vertex_indices]
        patch_normals = normals[patch_vertex_indices]

        patches.append((patch_vertices, patch_normals))

    return patches

def split_into_patches_adaptive(vertices, faces, num_patches=3, target_points_per_patch=1000):
    """
    Automatically adapts patch radius to get roughly consistent patch sizes.

    Args:
        vertices: (N, 3) torch.FloatTensor
        faces: (M, 3) torch.LongTensor (1-based indices!)
        num_patches: int, number of patches to sample
        target_points_per_patch: int, target number of points per patch

    Returns:
        patch_list: list of tuples (patch_vertices, patch_faces)
    """
    N = vertices.shape[0]
    patches = []

    # Estimate appropriate patch radius based on point density
    # Calculate bounding box to understand data scale
    bbox_min = vertices.min(dim=0)[0]
    bbox_max = vertices.max(dim=0)[0]
    bbox_diagonal = torch.norm(bbox_max - bbox_min)

    # Estimate point density
    volume_estimate = torch.prod(bbox_max - bbox_min)
    point_density = N / volume_estimate

    # Calculate radius to get target number of points
    # Assuming roughly spherical distribution
    target_volume = target_points_per_patch / point_density
    patch_radius = (3 * target_volume / (4 * torch.pi)) ** (1 / 3)

    # Clamp radius to reasonable bounds
    min_radius = bbox_diagonal * 0.01  # At least 1% of diagonal
    max_radius = bbox_diagonal * 0.3  # At most 30% of diagonal
    patch_radius = torch.clamp(patch_radius, min_radius, max_radius)

    print(f"Auto-calculated patch radius: {patch_radius:.4f}")
    print(f"Bounding box diagonal: {bbox_diagonal:.4f}")

    # Randomly sample patch centers
    indices = torch.randperm(N)[:num_patches]
    centers = vertices[indices]

    for center in centers:
        # Find vertices within radius
        distances = torch.norm(vertices - center, dim=1)
        mask = distances < patch_radius
        patch_vertex_indices = torch.nonzero(mask).squeeze(1)
        patch_vertices = vertices[patch_vertex_indices]

        if patch_vertex_indices.numel() == 0:
            continue

        print(f"Patch has {patch_vertex_indices.numel()} points")

        # Rest of the face processing logic...
        face_mask = mask[faces - 1].all(dim=1)
        selected_faces = faces[face_mask]

        global_patch_ids = patch_vertex_indices + 1
        global_to_local = {int(g.item()): i + 1 for i, g in enumerate(global_patch_ids)}

        remapped_faces = []
        for face in selected_faces:
            f0, f1, f2 = face.tolist()
            if f0 in global_to_local and f1 in global_to_local and f2 in global_to_local:
                remapped_faces.append([
                    global_to_local[f0],
                    global_to_local[f1],
                    global_to_local[f2]
                ])

        if len(remapped_faces) == 0:
            continue

        patch_faces = torch.tensor(remapped_faces, dtype=torch.long)
        patches.append((patch_vertices, patch_faces))

    return patches


def split_into_patches_with_normals(vertices, normals, num_patches=3, patch_radius=0.05):
    """
    Splits the full mesh into local patches (vertices + normals),
    each covering a neighborhood around randomly sampled centers.

    Args:
        vertices: (N, 3) torch.FloatTensor
        normals:  (N, 3) torch.FloatTensor (precomputed per-vertex normals)
        num_patches: int, number of patches to sample
        patch_radius: float, radius for each patch

    Returns:
        patch_list: list of tuples (patch_vertices, patch_normals)
            - patch_vertices: (P, 3) torch.FloatTensor
            - patch_normals:  (P, 3) torch.FloatTensor
    """
    N = vertices.shape[0]
    patches = []

    indices = torch.randperm(N)[:num_patches]
    centers = vertices[indices]

    for center in centers:
        distances = torch.norm(vertices - center, dim=1)
        mask = distances < patch_radius
        patch_vertices = vertices[mask]
        patch_normals = normals[mask]

        if patch_vertices.shape[0] == 0:
            continue
        patches.append((patch_vertices, patch_normals))

    return patches
