import torch

import os
import torch

def load(folder_path):
    """
    Load all .obj files in a folder, extracting vertex positions and face indices.

    Returns:
        meshes: list of tuples (vertices: (N, 3) torch.FloatTensor, faces: (M, 3) torch.LongTensor)
    """
    meshes = []

    for filename in os.listdir(folder_path):
        if filename.endswith('.obj'):
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

            vertices_tensor = torch.tensor(vertices, dtype=torch.float32)
            faces_tensor = torch.tensor(faces, dtype=torch.long)
            meshes.append((vertices_tensor, faces_tensor))

    return meshes
