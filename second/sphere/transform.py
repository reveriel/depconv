##
# transformation
# from xyz to range iamge

import numpy as np
from spconv.utils import points_to_voxel_3d_sphere_np

# the resolution of the LiDAR is 0.09 dgree for 5Hz. At 10Hz, the resolution is around 0.1728 degree.
# Ideally, W comes out to be 520

def xyz2range(points):
    """ convert points to depth map
        devide evenly.
        for KITTI dataset.
    Args:
        points: numpy tensor of shape [N, 4],
            point cloud, N is the number of points

    Returns:
        numpy tensor of shape [5, 64, 512]
    """
    # print("points shape = ", points.shape)
    x = points[:, 0]  # -71~73
    y = points[:, 1]  # -21~53
    z = points[:, 2]  # -5~2.6
    intensity = points[:, 3]  # 0~1
    # convert xyz to theta-phi-r
    x2y2 = x * x + y * y
    r = np.sqrt(x2y2 + z * z)
    # r[r == 0] = 1e-6

    thetas = np.arccos(z / r)
    phis = np.arcsin(y / np.sqrt(x2y2))
    # plt.plot(phis, thetas)
    # plt.show()

    # print("min thetas = ", np.min(thetas))
    # print("max theats = ", np.max(thetas))
    # print("min phis = ", np.min(phis))
    # print("max phis = ", np.max(phis))
    delta_phi = np.radians(90./512.)
    # delta_phi = (np.max(phis) - np.min(phis)) / 511.
    # veldyne , resolution 26.8  vertical,
    delta_theta = np.radians(26.8/64)
    # delta_theta = (np.max(thetas) - np.min(thetas)) / 64.
    # + 32 to map theta = 90 deg to the center of image
    theta_idx = (((thetas - np.pi/2) / delta_theta) + 31).astype(int)

    # print("out of range count =", np.sum(theta_idx >= 64))
    theta_idx[theta_idx >= 64] = 63
    # '-'phis since the direction of increasing phi is oppsosite to y on image.
    phi_idx = (phis / delta_phi + 256).astype(int)

    H = 64
    W = 512
    C = 5
    range_map = np.zeros((C, H, W), dtype=float)
    range_map[0, theta_idx, phi_idx] = x
    range_map[1, theta_idx, phi_idx] = y
    range_map[2, theta_idx, phi_idx] = z
    range_map[3, theta_idx, phi_idx] = r
    range_map[4, theta_idx, phi_idx] = intensity
    return range_map


def xyz2range_v2(points, ith=0):
    """ convert points to depth map
        devide evenly.
        for KITTI dataset.
        assume The input pointcloud has remove points outside picture
    Args:
        points: numpy tensor of shape [N, 4],
            point cloud, N is the number of points

    Returns:
        numpy tensor of shape [5, 64, 512]
    """
    x = points[:, 0]  # -71~73
    y = points[:, 1]  # -21~53
    z = points[:, 2]  # -5~2.6
    intensity = points[:, 3]  # 0~1

    #   plot_point_cloud_scatter_2d(x,y)
    #   plot_point_cloud_scatter_3d(x,y,z)

    x2y2 = x * x + y * y
    r = np.sqrt(x2y2 + z * z)
    # phi
    # arctan2 , and arcsin, almost the same result
    phi = -np.arctan2(y, x)
    # phi_p = np.arcsin(-y / np.sqrt(x ** 2 + y ** 2 ))
    # print("x rad diff", phi, phi_p, np.sum(phi - phi_p))
    angle_diff = np.diff(phi)
    #  plt.plot(np.diff(phi))
    #  plt.show()

    threshold_angle = np.radians(10)  # huristic
    angle_diff = np.hstack((angle_diff, 0.0001))  # append one
    # new row when diff bigger than threashold
    angle_diff_mask = angle_diff > threshold_angle
    theta_idx = np.cumsum(angle_diff_mask)
    theta_idx += 20  # shift down for 20 pixels. XXX: bad practice.
    theta_idx[theta_idx >= 64] = 63
    # print("theta max min", theta_idx.max(), theta_idx.min())
    delta_phi = np.radians(90. / 512.)
    # delta_phi = np.radians((x.max() - x.min()) / 500.) # doenst work. many missing dots horiizontally
    phi_idx = np.floor((phi) / delta_phi + 256).astype(int)

    # phi_idx -= np.min(phi_idx) # translate to positive
    phi_idx[phi_idx >= 512] = 511
    # x_max = int(360.0 / h_res) #+ 1  # 投影后图片的宽度
    x_max = 512
    # 可能有 data loss， 有些数据点被覆盖了。
    depth_map = np.zeros((5, 64, 512), dtype=float)  # +255
    depth_map[0, theta_idx, phi_idx] = x
    depth_map[1, theta_idx, phi_idx] = y
    depth_map[2, theta_idx, phi_idx] = z
    depth_map[3, theta_idx, phi_idx] = r
    depth_map[4, theta_idx, phi_idx] = intensity
    return depth_map


class VoxelGeneratorV3:
    def __init__(self,
                 voxel_size,
                 point_cloud_range,
                 max_num_points,
                 max_voxels=30000):
        point_cloud_range = np.array(point_cloud_range, dtype=np.float32)
        voxel_size = np.array(voxel_size, dtype=np.float32)
        grid_size = np.array([512, 512, 63])  # r, phi, theta
        voxelmap_shape = np.array([512,512,63])[::-1]

        self._max_voxels = max_voxels
        self._max_ponts = 1

        self._coor_to_voxelidx = np.full(voxelmap_shape, -1, dtype=np.int32)
        self._voxel_size = voxel_size
        self._point_cloud_range = point_cloud_range
        self._max_num_points = max_num_points
        self._max_voxels = max_voxels
        self._grid_size = grid_size
        # self._full_mean = full_mean

    def generate(self, points, max_voxels):
        res = points_to_voxel_v2(
            points,
            self._voxel_size[[0]],
            self._point_cloud_range[[0,3]],
            self._coor_to_voxelidx,
            self._max_num_points, max_voxels or self._max_voxels)

        for k, v in res.items():
            if k != "voxel_num":
                res[k] = v[:res["voxel_num"]]
        return res

    def generate_multi_gpu(self, points, max_voxels=None):
        res = points_to_voxel_v2(points,
            self._voxel_size[[0]],
            self._point_cloud_range[[0,3]],
            self._coor_to_voxelidx,
            self._max_num_points, max_voxels or self._max_voxels,
            pad_output=True
            )
        # print("res voels .shape  = ", res["voxels"].shape)
        return res

    @property
    def voxel_size(self):
        return self._voxel_size

    @property
    def max_num_points_per_voxel(self):
        return self._max_num_points

    @property
    def point_cloud_range(self):
        return self._point_cloud_range

    @property
    def grid_size(self):
        return self._grid_size


def points_to_voxel_v2(points,
                    voxel_size,
                    coors_range,
                    coor_to_voxelidx,
                    max_points=1,
                    max_voxels=30000,
                    pad_output=False):

    # voxelmap_shape = np.array([512,512,64])
    num_points_per_voxel = np.zeros(shape=(max_voxels,), dtype=np.int32)
    voxels = np.zeros(
        shape=(max_voxels, max_points, points.shape[-1]), dtype=points.dtype)

    voxel_point_mask = np.zeros(
        shape=(max_voxels, max_points), dtype=points.dtype)

    coors = np.zeros(shape=(max_voxels, 3), dtype=np.int32)
    res = {
        "voxels": voxels,
        "coordinates": coors,
        "num_points_per_voxel": num_points_per_voxel,
        "voxel_point_mask": voxel_point_mask,
    }

    voxel_num = points_to_voxel_3d_sphere_np(
        points, voxels, voxel_point_mask, coors,
        num_points_per_voxel, coor_to_voxelidx, voxel_size.tolist(),
        coors_range.tolist(), max_points, max_voxels)

    res["voxel_num"] = voxel_num
    res["voxel_point_mask"] = res["voxel_point_mask"].reshape(
        -1, max_points, 1)
    return res