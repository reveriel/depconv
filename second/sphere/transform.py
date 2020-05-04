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

    """
    print("points shape = ", points.shape)
    x = points[:, 0]  # -71~73
    y = points[:, 1]  # -21~53
    z = points[:, 2]  # -5~2.6
    intensity = points[:, 3]  # 0~1
    # convert xyz to theta-phi-r
    x2y2 = x* x + y*y
    distance = np.sqrt(x2y2 + np.square(z))
#     distance[distance == 0] = 1e-6
    thetas = np.arcsin(z / distance)
    phis = np.arcsin(-y / np.sqrt(x2y2))
    # print("min thetas = ", np.min(thetas))
    # print("max theats = ", np.max(thetas))
    # print("min phis = ", np.min(phis))
    # print("max phis = ", np.max(phis))
    delta_phi = np.radians(90./512.)
#     delta_phi = (np.max(phis) - np.min(phis)) / 511.
    # veldyne , resolution 26.8  vertical,
    delta_theta = np.radians(26.8/64)
#     delta_theta = (np.max(thetas) - np.min(thetas)) / 64.
    theta_idx = ((thetas - np.min(thetas)) / delta_theta).astype(int)
    phi_idx = ((phis - np.min(phis)) / delta_phi).astype(int)
    print("theta_idx = ", theta_idx)
    print("phi_idx = ", phi_idx)

    H = 64
    W = 512
    C = 5
    range_map = np.zeros((C,H,W), dtype=float)
    range_map[0, theta_idx,phi_idx] = x
    range_map[1, theta_idx,phi_idx] = y
    range_map[2, theta_idx,phi_idx] = z
    range_map[3, theta_idx,phi_idx] = distance
    range_map[4, theta_idx,phi_idx] = intensity
    return range_map


def xyz2range_v2(points):
    """
    """
    v_res=26.9/64,
    h_res=90./512.
    x = points[:, 0]  # -71~73
    y = points[:, 1]  # -21~53
    z = points[:, 2]  # -5~2.6
    intensity = points[:, 3]  # 0~1

    x2y2 = np.sqrt(np.square(x) + np.square(y))
    distance = np.sqrt(x2y2 + np.square(z))
    # phi
    # arctan2 , and arcsin, almost the same result
    phi = -np.arctan2(y, x)
    # phi_p = np.arcsin(-y / np.sqrt(x ** 2 + y ** 2 ))
    # print("x rad diff", phi, phi_p, np.sum(phi - phi_p))
    angle_diff = np.diff(phi)
    threshold_angle = np.radians(10)  # huristic
    angle_diff = np.hstack((angle_diff, 0.0001)) # append one
    # new row when diff bigger than threashold
    angle_diff_mask = angle_diff > threshold_angle
    theta_idx = np.cumsum(angle_diff_mask)
    theta_idx[theta_idx >= 64] = 63
    # print("theta max min", theta_idx.max(), theta_idx.min())
    delta_phi = np.radians(90./512.)
    # delta_phi = np.radians((x.max() - x.min()) / 500.) # doenst work. many missing dots horiizontally
    phi_idx = np.floor((phi + np.pi/2) / delta_phi).astype(int)
#    phi_idx -= np.min(phi_idx) # translate to positive
    phi_idx[phi_idx >= 512] = 511
    # x_max = int(360.0 / h_res) #+ 1  # 投影后图片的宽度
    x_max = 512
    if visualize:
        depth_gray = np.interp(distance, (distance.min(), distance.max()), (20,255))
        depth_image = np.zeros((64, 512, 1))
        depth_image[theta_idx, phi_idx, 0] = depth_gray
#        imageio.imwrite(fig_path + 'range_map_v2_{:d}.jpg'.format(ith),
 #           depth_image.astype(np.uint8))
 # 可能有 data loss， 有些数据点被覆盖了。
    depth_map = np.zeros((5, 64, 512), dtype=float) #+255
    depth_map[0, theta_idx, phi_idx] = x
    depth_map[1, theta_idx, phi_idx] = y
    depth_map[2, theta_idx, phi_idx] = z
    depth_map[3, theta_idx, phi_idx] = distance
    depth_map[4, theta_idx, phi_idx] = intensity
    return depth_map