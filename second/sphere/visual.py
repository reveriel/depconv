import numpy as np
import matplotlib.pyplot as plt
import imageio

# This import registers the 3D projection, but is otherwise unused.
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
from second.sphere.conv import depth_to_3D


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

def save_2d_jpg(feature, name:str):
    """
    save a 2d feature map to image file
    depth : tensor of shape (H, W)
    """
    save_depth_jpg(feature, name)


depth_map_bev_count = 0
def depthmap_bev(feature, depth, D=0):
    """
    aruguments:
        feature: tensor of shape (B,C,H,W)
        depth: (B,H,W)
        D: the max of 'depth', if not given, it will guess
    save image file
    """
    global depth_map_bev_count
    f3d = depth_to_3D(feature, depth, D=D)
    B, C, D, H, W = f3d.shape
    for C_i in range(C):
        f3d_piece = f3d[0][C_i]
        # sum over depth, D,H,W
        f2d = f3d_piece.sum(dim=1)
        save_2d_jpg(f2d, "fmap-%d-%d.jpg" % (depth_map_bev_count , C_i))
        depth_map_bev_count +=1