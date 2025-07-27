import open3d as o3d
import numpy as np
import torch
import matplotlib.pyplot as plt

import open3d as o3d
import torch

def visualize_patches(patch_vertices, colorize=True):
    """
    Visualize a list of point cloud patches using Open3D.

    Args:
        patch_vertices: list of patches (each patch is a set of 3D points)
        colorize: bool, whether to color each patch differently
    """
    geometries = []

    for i, patch in enumerate(patch_vertices):
        # Ensure the tensor is on CPU and convert to numpy
        points_np = patch.cpu().numpy()

        # Create Open3D point cloud
        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(points_np)

        # Assign different color to each patch if desired
        if colorize:
            color = torch.rand(3).tolist()
            pc.paint_uniform_color(color)

        geometries.append(pc)

    # Visualize all point clouds together
    o3d.visualization.draw_geometries(geometries)
