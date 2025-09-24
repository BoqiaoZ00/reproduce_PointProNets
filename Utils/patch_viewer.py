import open3d as o3d
import numpy as np
import torch
import matplotlib.pyplot as plt
import torch
import plotly.graph_objects as go
import numpy as np


def visualize_patches(patch_vertices, colorize=True):
    """
    Interactive 3D visualization with full rotation, zoom, pan capabilities.
    """
    fig = go.Figure()

    # Color palette
    colors = ['red', 'green', 'blue', 'yellow', 'magenta', 'cyan', 'orange', 'purple', 'brown', 'pink']

    for i, patch in enumerate(patch_vertices):
        points_np = patch.cpu().numpy()

        print(f"Patch {i} shape: {points_np.shape}")

        if points_np.shape[1] != 3:
            points_np = points_np.reshape(-1, 3)

        # Sample points if too many (for performance)
        if len(points_np) > 3000:
            indices = np.random.choice(len(points_np), 3000, replace=False)
            points_np = points_np[indices]

        color = colors[i % len(colors)] if colorize else 'blue'

        fig.add_trace(go.Scatter3d(
            x=points_np[:, 0],
            y=points_np[:, 1],
            z=points_np[:, 2],
            mode='markers',
            marker=dict(
                size=2,
                color=color,
                opacity=0.8
            ),
            name=f'Patch {i}',
            showlegend=True
        ))

    fig.update_layout(
        title='Interactive Point Cloud Patches',
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.5)
            )
        ),
        width=1000,
        height=800
    )

    fig.show()


# def visualize_patches(patch_vertices, colorize=True):
#     """
#     Visualize patches using matplotlib 3D scatter plot.
#     """
#     fig = plt.figure(figsize=(10, 8))
#     ax = fig.add_subplot(111, projection='3d')
#
#     colors = plt.cm.tab10(np.linspace(0, 1, len(patch_vertices))) if colorize else ['blue'] * len(patch_vertices)
#
#     for i, patch in enumerate(patch_vertices):
#         points_np = patch.cpu().numpy()
#
#         # Ensure proper shape
#         if points_np.ndim != 2 or points_np.shape[1] != 3:
#             points_np = points_np.reshape(-1, 3)
#
#         # Filter out invalid points
#         valid_mask = np.isfinite(points_np).all(axis=1)
#         points_np = points_np[valid_mask]
#
#         if len(points_np) > 0:
#             ax.scatter(points_np[:, 0], points_np[:, 1], points_np[:, 2],
#                        c=[colors[i]], label=f'Patch {i}', alpha=0.6, s=20)
#
#     ax.set_xlabel('X')
#     ax.set_ylabel('Y')
#     ax.set_zlabel('Z')
#     ax.legend()
#     plt.title('Point Cloud Patches')
#     plt.show()


def visualize_heightmap(heightmap, title='Heightmap'):
    """
    Visualize a single heightmap tensor (2D) as a grayscale image.

    Args:
        heightmap: torch.Tensor or numpy.ndarray of shape (k, k)
        title: Optional string for plot title
    """
    if isinstance(heightmap, torch.Tensor):
        heightmap = heightmap.detach().cpu().numpy()

    plt.figure(figsize=(5, 5))
    plt.imshow(heightmap, cmap='gray', origin='lower')
    plt.title(title)
    plt.colorbar(label='Height')
    plt.axis('off')
    plt.tight_layout()
    plt.show()


def visualize_patch_with_normal(patch_vertices, normal_vector, patch_name="Patch", point_size=3, normal_scale=0.1):
    """
    Interactive 3D visualization of a single patch with its normal vector.

    Args:
        patch_vertices: Tensor of shape (N, 3) - the point cloud patch
        normal_vector: Tensor of shape (3,) - the normal vector
        patch_name: Name for the legend
        point_size: Size of the points in the visualization
        normal_scale: Scale factor for the normal vector (for visibility)
    """
    fig = go.Figure()

    # Convert to numpy if needed
    if isinstance(patch_vertices, torch.Tensor):
        points_np = patch_vertices.detach().cpu().numpy()
    else:
        points_np = patch_vertices.copy()

    if isinstance(normal_vector, torch.Tensor):
        normal_np = normal_vector.detach().cpu().numpy()
    else:
        normal_np = normal_vector.copy()

    # Ensure correct shapes
    if points_np.shape[1] != 3:
        points_np = points_np.reshape(-1, 3)

    # Calculate patch center (average point)
    patch_center = points_np.mean(axis=0)

    # Scale normal vector for better visibility
    scaled_normal = normal_np * normal_scale

    # End point of normal vector
    normal_end = patch_center + scaled_normal

    # Sample points if too many (for performance)
    if len(points_np) > 3000:
        indices = np.random.choice(len(points_np), 3000, replace=False)
        points_np = points_np[indices]

    # Plot the point cloud patch
    fig.add_trace(go.Scatter3d(
        x=points_np[:, 0],
        y=points_np[:, 1],
        z=points_np[:, 2],
        mode='markers',
        marker=dict(
            size=point_size,
            color='blue',
            opacity=0.7
        ),
        name=f'{patch_name} Points',
        showlegend=True
    ))

    # Plot the normal vector as an arrow
    fig.add_trace(go.Scatter3d(
        x=[patch_center[0], normal_end[0]],
        y=[patch_center[1], normal_end[1]],
        z=[patch_center[2], normal_end[2]],
        mode='lines+markers',
        line=dict(
            color='red',
            width=8
        ),
        marker=dict(
            size=6,
            color='red'
        ),
        name='Normal Vector',
        showlegend=True
    ))

    # Add a sphere at the base of the normal vector for better visibility
    fig.add_trace(go.Scatter3d(
        x=[patch_center[0]],
        y=[patch_center[1]],
        z=[patch_center[2]],
        mode='markers',
        marker=dict(
            size=8,
            color='green',
            symbol='circle'
        ),
        name='Patch Center',
        showlegend=True
    ))

    fig.update_layout(
        title=f'{patch_name} with Normal Vector<br>Normal: [{normal_np[0]:.3f}, {normal_np[1]:.3f}, {normal_np[2]:.3f}]',
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.5)
            ),
            aspectmode='data'  # Equal aspect ratio
        ),
        width=900,
        height=700
    )

    fig.show()

    # Print some statistics
    print(f"Patch Statistics:")
    print(f"  Number of points: {len(points_np)}")
    print(f"  Normal vector: [{normal_np[0]:.3f}, {normal_np[1]:.3f}, {normal_np[2]:.3f}]")
    print(f"  Normal magnitude: {np.linalg.norm(normal_np):.3f}")
    print(f"  Patch center: [{patch_center[0]:.3f}, {patch_center[1]:.3f}, {patch_center[2]:.3f}]")