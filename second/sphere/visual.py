import numpy as np
import matplotlib.pyplot as plt
import imageio

# This import registers the 3D projection, but is otherwise unused.
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import


def depth_from_feature_np(feature, k):
    """
    Args:
        feature: numpy array of shape [5, 64, 512], C, H, W,
            the 3th channel is depth channel

    Return:
        depth map, of shape [64, 512]  scaled to  [0 to k],
        for grey scale plotting
    """
    r = feature[3, :, :]
    depth = ((r - r.min()) * (k) / (r.max() - r.min()))
    return depth


def plot_point_cloud_scatter_2d(x, y):
    """
    given a point cloud, plot the scattered map
    Args:
        x : numpy array of shape [N], x coordinates
        y : numpy array of shape [N], y coordinates
    """
    plt.scatter(x, y, linewidths=0.1, marker='.')
    plt.show()


def plot_point_cloud_scatter_3d(x, y, z):
    """
    given a point cloud, plot the scattered map
    Args:
        x : numpy array of shape [N], x coordinates
        y : numpy array of shape [N], y coordinates
        z : numpy array of shape [N], z coordinates
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z, linewidths=1)
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    plt.show()


def save_depth_jpg(depth, path: str):
    """
    save depth map to image file
    Args:
        depth : np tensor of shape (H, W)
        path
    """
    depth = depth.cpu().numpy()
    gray = np.interp(depth, (depth.min(), depth.max()), (50, 255))
    H, W = depth.shape
    imageio.imwrite(path, gray.astype(np.uint8))
