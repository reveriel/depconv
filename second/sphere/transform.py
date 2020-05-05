##
# transformation
# from xyz to range iamge

import numpy as np

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
    phi_idx = (-phis / delta_phi + 256).astype(int)

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
