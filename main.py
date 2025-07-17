import sys
import torch
import Utils.data_loader as data_loader
import HeightmapGenerator as HGN
import HeightmapDenoiser as HDN
from Utils.ground_truth_loader import compute_gt_heightmap, compute_gt_normals
from Utils.patch_splitter import split_into_patches


def main():
    X = data_loader.load("./data")
    print(len(X)) # should be 4 (.obj files)
    print(len(X[0])) # should be 2 ([0] is vertices, [1] is faces)
    print(X[0][0].shape)
    ngt = torch.mean(compute_gt_normals(X[0][0], X[0][1]), dim=0) # the global ngt for the first item
    print(ngt.shape)
    patches = split_into_patches(X[0][0], X[0][1])
    print(len(patches))
    print(patches[0][0].shape) # patches[0] is the first patch, patches[0][0] is the vertices in this patch

    Hgt, ngt = compute_gt_heightmap(patches[0][0], patches[0][1], ngt) # try a patch with pre_computed ngt
    print(Hgt.shape)
    print(ngt.shape)

    # test project_points_to_heightmap_exact
    # n = HGN.get_normal(X)
    # HGN.project_points_to_heightmap_exact(X, n)

    # Training Process
    # Step 1: get Ygt, Hgt, ngt as above (randomly choose +- for ngt)
    # get global ngt for each item -> split into patches -> compute Hgt for each patch with the pre_computed global ngt

    # Step 2: train HDN first (using ground truth plane parameters)

    # Step 3: train HGN

if __name__ == "__main__":
    main()
