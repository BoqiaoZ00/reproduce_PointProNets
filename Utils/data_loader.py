import torch

import os
import torch

def load(folder_path, device=None):
    """
    Load all .obj files in a folder, extracting vertex positions and face indices.

    Returns:
        meshes: list of tuples (vertices: (N, 3) torch.FloatTensor, faces: (M, 3) torch.LongTensor)
    """
    if device is None:
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    # 1. List all .obj files
    obj_files = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.obj'):
            obj_files.append(filename)

    obj_files.sort()  # This sorts alphabetically by default
    meshes = []

    for filename in obj_files:
        vertices = []
        faces = []

        with open(os.path.join(folder_path, filename), 'r') as f:
            for line in f:
                if line.startswith('v '):
                    parts = line.strip().split()
                    vertex = [float(parts[1]), float(parts[2]), float(parts[3])]
                    vertices.append(vertex)
                elif line.startswith('f '):
                    parts = line.strip().split()
                    # OBJ indices are 1-based, so subtract 1
                    face = [int(p.split('/')[0]) for p in parts[1:4]]
                    faces.append(face)

        vertices_tensor = torch.tensor(vertices, dtype=torch.float32, device=device)
        faces_tensor = torch.tensor(faces, dtype=torch.long, device=device)
        meshes.append((vertices_tensor, faces_tensor))

    return meshes

import torch

def load_thimble(data_path, device=None):
    """
    Loads vertex positions (v) and vertex normals (vn).

    This function assumes the .obj file is structured so that
    the i-th 'v' line corresponds to the i-th 'vn' line, as
    confirmed by the face data (e.g., 'f 8097/../8097').

    Returns:
        vertices: (N, 3) torch.FloatTensor of vertex positions
        normals: (N, 3) torch.FloatTensor of vertex normals
    """
    vertices = []
    normals = []

    with open(data_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('v '):
                parts = line.split()
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif line.startswith('vn '):
                parts = line.split()
                normals.append([float(parts[1]), float(parts[2]), float(parts[3])])

    vertices_tensor = torch.tensor(vertices, dtype=torch.float32, device=device)
    normals_tensor = torch.tensor(normals, dtype=torch.float32, device=device)

    # Sanity check to ensure the lists are the same length
    if vertices_tensor.shape[0] != normals_tensor.shape[0]:
        print(f"Warning: Vertex count ({vertices_tensor.shape[0]}) "
              f"does not match normal count ({normals_tensor.shape[0]}).")
        # You might want to raise an error here if they *must* match
        # raise ValueError("Vertex and normal counts do not match!")

    return vertices_tensor, normals_tensor


def load_with_normals(folder_path, device=None):
    """
    Load all .vn files in a folder.
    Each line format: v x y z nx ny nz
    Returns:
        meshes: list of tuples (vertices: (N,3), normals: (N,3)) torch.FloatTensors
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    meshes = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".vn"):
            verts = []
            norms = []
            with open(os.path.join(folder_path, filename), "r") as f:
                for line in f:
                    if line.startswith("v "):
                        parts = line.strip().split()
                        if len(parts) != 7:
                            continue
                        x, y, z, nx, ny, nz = map(float, parts[1:])
                        verts.append([x, y, z])
                        norms.append([nx, ny, nz])
            if verts:
                vertices_tensor = torch.tensor(verts, dtype=torch.float32, device=device)
                normals_tensor = torch.tensor(norms, dtype=torch.float32, device=device)
                meshes.append((vertices_tensor, normals_tensor))
    return meshes