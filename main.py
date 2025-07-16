import sys
import torch
import Utils.data_loader as data_loader
import HeightmapGenerator as HGN
from Utils.ground_truth_loader import compute_gt_heightmap


def main():
    X = data_loader.load()
    print(X)
    Hgt = compute_gt_heightmap(X[0], torch.tensor([[1,2,3], [1,2,4]]))
    print(Hgt.shape)
    # n = HGN.get_normal(X)
    # HGN.project_points_to_heightmap_exact(X, n)

    # Step 1: get Ygt, Hgt, ngt(randomly choose +-)

    # Step 2: train HDN first (using ground truth plane parameters)

    # Step 3: train HGN

if __name__ == "__main__":
    main()
