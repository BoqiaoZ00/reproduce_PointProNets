import torch

def split_into_patches(vertices, faces, num_patches=3, patch_radius=10.0):
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
