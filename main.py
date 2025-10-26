import bz2
import os
import sys
from io import BytesIO
from pathlib import Path

import imageio.v2 as imageio
from PIL import Image
import torch
import torch.nn.functional as F
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from torch import Tensor

import Utils.data_loader as data_loader
import HeightmapGenerator as HGN
import HeightmapDenoiser as HDN
from HeightmapGenerator import project_points_to_heightmap_exact
from Utils.data_loader import load_thimble
from Utils.ground_truth_loader import compute_gt_heightmap, compute_gt_normals
from Utils.patch_splitter import split_into_patches, split_into_patches_adaptive, split_into_patches_with_normals, \
    split_thimble_into_patches
from Utils.patch_viewer import visualize_heightmap, visualize_patches, visualize_patch_with_normal

# def main():
#     print(torch.backends.mps.is_available())  # Should return True
#     print(torch.device('mps'))  # Should print 'mps' (M1/M2 GPU)
#
#     X = data_loader.load("./data_density_normalized", device=torch.device('cpu'))
#     print(len(X)) # should be 15 (.obj files)
#     print(len(X[1])) # should be 2 ([0] is vertices, [1] is faces)
#     print(X[1][0].shape)
#
#     patches = split_into_patches(X[5][0], X[5][1], num_patches=40, patch_radius=0.1)  # patches for the first item
#     print(len(patches))
#
#     # 1. Unpack patches into two lists: one for vertices, one for faces
#     vertices_list = []
#     faces_list = []
#
#     for patch in patches:
#         verts, faces = patch
#         print(len(verts))
#         vertices_list.append(verts)
#         faces_list.append(faces)
#
#     # 2. Compute normal for each patch
#     normals_list = []
#     for verts, faces in zip(vertices_list, faces_list):
#         # Compute per-face normals and average to get patch-level normal
#         per_face_normals = compute_gt_normals(verts, faces)
#         normal = torch.mean(per_face_normals, dim=0)
#         normal = F.normalize(normal, dim=0)  # ensure unit norm
#         normals_list.append(normal)
#
#     # Test viewer
#     point_patches = [v for v, _ in patches]
#     visualize_patches(point_patches)

# Test ground_truth_loader.compute_gt_heightmap
    # Hgt, ngt = compute_gt_heightmap(patches[0][0], patches[0][1], ngt) # try a patch with pre_computed ngt
    # # This ngt should be unchanged because it's pre_computed global ngt
    # print(Hgt.shape)
    # print(ngt.shape)

    # test HeightmapGenerator.project_points_to_heightmap_exact
    # point_patches = torch.tensor([
    #     [
    #         [0.0, 1.0, 1.0],  # point A
    #         [1.0, 1.0, 1.0],  # point B
    #         [1.0, 0.0, 1.0],  # point C
    #         [1.0, 0.0, 1.0],
    #         [-1.0, 0.0, 1.0],
    #         [-1.0, -0.5, 1.0],
    #         [-1.0, -2.0, 1.0],
    #         [1.0, -2.0, 1.0],
    #         [1.0, 0.5, 1.0]
    #     ]
    # ])  # shape: (3, 3)
    # ngt = torch.tensor([0.0, 0.0, 1.0])

    # 3. Project all patches to heightmaps (patched)
    # r = 1.0
    # heightmap_list = HGN.project_points_to_heightmap_exact(
    #     patch_list=vertices_list,
    #     normals=normals_list,
    #     r=r
    # )
    # for i, hmap in enumerate(heightmap_list):
    #     visualize_heightmap(hmap, title=f'Patch {i}')
    #
    # # OR as below
    # heightmap_list = []
    # for verts, faces in zip(vertices_list, faces_list):
    #     # Compute per-face normals and average to get patch-level normal
    #     heightmap,_ = compute_gt_heightmap(verts, faces, r=r)
    #     heightmap_list.append(heightmap)
    #
    # for i, hmap in enumerate(heightmap_list):
    #     visualize_heightmap(hmap, title=f'Patch {i}')

    # Training Process
    # Step 1: get Ygt, Hgt, ngt as above (randomly choose +- for ngt)
    # get global ngt for each item -> split into patches -> compute Hgt for each patch with the pre_computed global ngt

    # Step 2: train HDN first (using ground truth plane parameters)

    # Step 3: train HGN

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from Utils import data_loader
from Utils.patch_splitter import split_into_patches
from Utils.preprocessing.normalize import normalize_mesh

import matplotlib.pyplot as plt
import numpy as np


@torch.no_grad()
def add_point_spikes_along_normal(
    points: torch.Tensor,  # (P,3)
    patch_normal: torch.Tensor,  # (3,)
    *,
    num_points: int = 30,
    fixed_height: float = 5.0,  # fixed spike height in world units
    seed: int = None,
) -> torch.Tensor:
    """
    Make sparse 'obvious' outliers by moving random points along the patch normal.
    Uses a fixed spike height instead of relative to patch z-range.
    """
    device, dtype = points.device, points.dtype
    g = torch.Generator(device=device)
    if seed is not None:
        g.manual_seed(seed)

    P = points.size(0)
    if P == 0:
        return points.clone()

    # Normalize the patch normal
    n = patch_normal / (patch_normal.norm() + 1e-8)

    # Which points to spike?
    take = min(num_points, P)
    idx = torch.randperm(P, generator=g, device=device)[:take]

    # Build spikes (± fixed_height, with small randomness)
    rand_scale = 0.7 + 0.6 * torch.rand(take, generator=g, device=device)  # ~[0.7,1.3]
    signs = torch.where(torch.rand(take, generator=g, device=device) < 0.5, -1.0, 1.0)
    delta = fixed_height * rand_scale * signs  # (take,)

    noisy = points.clone()
    noisy[idx] = noisy[idx] + delta.unsqueeze(-1) * n  # move along the normal
    return noisy


@torch.no_grad()
def add_small_isotropic_jitter(
        points: torch.Tensor,
        *,
        sigma_rel_radius: float = 0.003,  # ~0.3% of patch radius (very mild)
        patch_radius: float = 0.15,
        seed: int = None,
) -> torch.Tensor:
    """
    Optional gentle jitter on *all* points (simulates sensor noise).
    """
    device, dtype = points.device, points.dtype
    sigma = sigma_rel_radius * patch_radius
    g = torch.Generator(device=device)
    if seed is not None:
        g.manual_seed(seed)
    try:
        # Preferred: randn with explicit size supports generator broadly
        noise = torch.randn(points.shape, device=device, dtype=dtype, generator=g) * sigma
    except TypeError:
        # Very old versions: fall back to manual seeding (global)
        print("fall back to manual seeding (global)")
        cpu = (device.type == "cpu")
        prev_state = torch.random.get_rng_state()
        if cpu:
            torch.manual_seed(seed)
        else:
            torch.cuda.manual_seed_all(seed)
        noise = torch.randn(points.shape, device=device, dtype=dtype) * sigma
        # restore RNG state (best-effort; CPU-only shown)
        torch.random.set_rng_state(prev_state)
    return points + noise

def plot_heightmap_2d(heightmap, title="Heightmap", cmap='viridis'):
    """
    Simple 2D heatmap visualization
    """
    plt.figure(figsize=(8, 6))
    plt.imshow(heightmap.detach().cpu().numpy(), cmap=cmap)
    plt.colorbar(label='Height')
    plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    plt.show()

from scipy.ndimage import gaussian_filter

def smooth_heightmap_numpy(tensor: torch.Tensor, sigma: float = 1.0) -> torch.Tensor:
    # Convert to NumPy
    np_array = tensor.cpu().numpy()
    # Apply Gaussian filter
    smoothed_np = gaussian_filter(np_array, sigma=sigma)
    # Convert back to torch
    return torch.from_numpy(smoothed_np).to(tensor.device)


def main():
    def save_obj(vertices, faces, filepath):
        """
        Save mesh as .obj with 'v' and 'f' lines.
        """
        with open(filepath, "w") as f:
            # Write vertices
            for v in vertices:
                f.write(f"v {v[0].item()} {v[1].item()} {v[2].item()}\n")

            # Write faces (faces are already 1-based)
            for face in faces:
                f.write(f"f {face[0].item()} {face[1].item()} {face[2].item()}\n")

    def load_obj_folder(folder_path, device=None):
        """
        Load all .obj files in a folder, extracting vertices and faces.
        """
        if device is None:
            device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

        meshes = []
        for filename in os.listdir(folder_path):
            if filename.endswith(".obj"):
                vertices, faces = [], []
                with open(os.path.join(folder_path, filename), "r") as f:
                    for line in f:
                        if line.startswith("v "):
                            parts = line.strip().split()
                            vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
                        elif line.startswith("f "):
                            parts = line.strip().split()
                            face = [int(p.split("/")[0]) for p in parts[1:4]]
                            faces.append(face)

                vertices_tensor = torch.tensor(vertices, dtype=torch.float32, device=device)
                faces_tensor = torch.tensor(faces, dtype=torch.long, device=device)
                meshes.append((vertices_tensor, faces_tensor))
        return meshes

    def visualize_mesh(vertices, faces, title="Mesh"):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

        verts_np = vertices.cpu().numpy()
        faces_np = faces.cpu().numpy() - 1  # convert to 0-based for indexing

        mesh_tris = verts_np[faces_np]
        ax.add_collection3d(
            Poly3DCollection(mesh_tris, facecolor="lightblue", edgecolor="k", linewidths=0.1, alpha=0.8)
        )

        ax.scatter(verts_np[:, 0], verts_np[:, 1], verts_np[:, 2], s=1, c="r", alpha=0.3)

        ax.set_title(title)
        ax.set_box_aspect([1, 1, 1])
        plt.show()

    # input_dir = "./data"
    # output_dir = "./data_normalized"
    # Path(output_dir).mkdir(parents=True, exist_ok=True)
    #
    # # Load all meshes
    # meshes = data_loader.load(input_dir, device=torch.device("cpu"))
    # print(f"Loaded {len(meshes)} meshes")
    #
    # for i, (vertices, faces) in enumerate(meshes):
    #     norm_faces = faces
    #     norm_vertices, params = normalize_mesh(vertices, method="unit_cube")
    #
    #     filename = f"mesh_{i:03d}.obj"
    #     filepath = Path(output_dir) / filename
    #     save_obj(norm_vertices, norm_faces, filepath)
    #
    #     print(f"Saved {filename} with {norm_vertices.shape[0]} verts, {norm_faces.shape[0]} faces")
    #     # visualize_mesh(vertices, faces, title="Mesh")
    #
    # print(f"\nAll normalized meshes saved to {output_dir}")

    # norm_folder = "./data"
    # norm_meshes =  data_loader.load(norm_folder, device=torch.device("cpu"))
    # print(f"Loaded {len(norm_meshes)} normalized meshes")
    #
    # armadillo = norm_meshes[14]
    # norm_vertices, norm_faces = armadillo[0], armadillo[1]
    # print(f"This object has {len(norm_vertices)} normalized vertices")
    # patch_lists = split_into_patches(norm_vertices, norm_faces, num_patches=500, patch_radius=10)
    # patch_vertices = [patch[0] for patch in patch_lists]
    # patch_faces = [patch[1] for patch in patch_lists]
    # visualize_patches(patch_vertices, colorize=True)
    #
    # normals = compute_gt_normals(patch_vertices[0], patch_faces[0])
    # normal_per_patch = torch.mean(normals, dim=0)
    #
    # visualize_patch_with_normal(patch_vertices[0], normal_per_patch)
    #
    # # add random noise (20% height) to points
    # patch_vertices_noisy = add_point_spikes_along_normal(
    #     points = patch_vertices[0], patch_normal = normal_per_patch, num_points=30, fixed_height=5, seed=0
    # )
    # # patch_vertices_noisy = add_small_isotropic_jitter(patch_vertices_noisy, sigma_rel_radius=0.003, patch_radius=10, seed=0)
    #
    #
    #
    # HN_clean = HGN.project_points_to_heightmap_test([patch_vertices[0]], [normal_per_patch], r=12)
    # plot_heightmap_2d(HN_clean[0])
    #
    # HN_clean_smooth = smooth_heightmap_numpy(HN_clean[0])
    # plot_heightmap_2d(HN_clean_smooth)
    #
    # HN_clean = HGN.project_points_to_heightmap_test([patch_vertices_noisy], [normal_per_patch], r=12)
    # plot_heightmap_2d(HN_clean[0])


    thimble_path = "glof_ball_sim_data/thimble_v+f.obj"
    # --- Example Usage ---
    vertices, normals = load_thimble(thimble_path)
    print(f"Loaded {vertices.shape[0]} vertices and {normals.shape[0]} normals.")
    print("Shapes:", vertices.shape, normals.shape)

    patch_lists = split_thimble_into_patches(vertices, normals, num_patches=500, patch_radius=0.1)
    patch_vertices = [patch[0] for patch in patch_lists]
    patch_normals = [patch[1] for patch in patch_lists]
    # visualize_patches(patch_vertices, colorize=True)

    per_patch_normals_list = [torch.mean(normals_in_patch, dim=0) for normals_in_patch in patch_normals]
    print(f"per_patch_normals_list: {len(per_patch_normals_list)}")

    visualize_patch_with_normal(patch_vertices[0], per_patch_normals_list[0])

    HN_clean = HGN.project_points_to_heightmap_test([patch_vertices[0]], [per_patch_normals_list[0]], r=0.1)
    plot_heightmap_2d(HN_clean[0])

    HN_clean_smooth = smooth_heightmap_numpy(HN_clean[0])
    plot_heightmap_2d(HN_clean_smooth)


if __name__ == "__main__":
    main()