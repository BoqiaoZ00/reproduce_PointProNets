import sys
import torch
import torch.nn.functional as F
import Utils.data_loader as data_loader
import HeightmapGenerator as HGN
import HeightmapDenoiser as HDN
from Utils.ground_truth_loader import compute_gt_heightmap, compute_gt_normals
from Utils.patch_splitter import split_into_patches
from Utils.patch_viewer import visualize_patches, visualize_heightmap


def main():
    print(torch.backends.mps.is_available())  # Should return True
    print(torch.device('mps'))  # Should print 'mps' (M1/M2 GPU)

    X = data_loader.load("./data", device=torch.device('cpu'))
    print(len(X)) # should be 4 (.obj files)
    print(len(X[0])) # should be 2 ([0] is vertices, [1] is faces)
    print(X[0][0].shape)

    patches = split_into_patches(X[0][0], X[0][1], num_patches=5)  # patches for the first item
    print(len(patches))

    # 1. Unpack patches into two lists: one for vertices, one for faces
    vertices_list = []
    faces_list = []

    for patch in patches:
        verts, faces = patch
        vertices_list.append(verts)
        faces_list.append(faces)

    # 2. Compute normal for each patch
    normals_list = []
    for verts, faces in zip(vertices_list, faces_list):
        # Compute per-face normals and average to get patch-level normal
        per_face_normals = compute_gt_normals(verts, faces)
        normal = torch.mean(per_face_normals, dim=0)
        normal = F.normalize(normal, dim=0)  # ensure unit norm
        normals_list.append(normal)

    # Test viewer
    # point_patches = [v for v, _ in patches]
    # visualize_patches(point_patches)

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
    heightmap_list = HGN.project_points_to_heightmap_exact(
        patch_list=vertices_list,
        normals=normals_list,
        r=10.0  # or any r you want
    )
    for i, hmap in enumerate(heightmap_list):
        visualize_heightmap(hmap, title=f'Patch {i}')

    # OR as below
    heightmap_list = []
    for verts, faces in zip(vertices_list, faces_list):
        # Compute per-face normals and average to get patch-level normal
        heightmap,_ = compute_gt_heightmap(verts, faces, r=10.0)
        heightmap_list.append(heightmap)

    for i, hmap in enumerate(heightmap_list):
        visualize_heightmap(hmap, title=f'Patch {i}')

    # Training Process
    # Step 1: get Ygt, Hgt, ngt as above (randomly choose +- for ngt)
    # get global ngt for each item -> split into patches -> compute Hgt for each patch with the pre_computed global ngt

    # Step 2: train HDN first (using ground truth plane parameters)

    # Step 3: train HGN

if __name__ == "__main__":
    main()
