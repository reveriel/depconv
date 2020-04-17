
from second.data.kitti_dataset import KittiDataset
import matplotlib.pyplot as plt
import matplotlib.collections as mcoll
import numpy as np
import imageio
import torch
import torch.nn.functional as F
from torch import nn
from typing import Dict, List, Tuple, Union

# from test_utils import params_grid, generate_sparse_data, TestCase
import unittest
from torchplus.tools import change_default_args
import math

fig_path = "output/"

# kitti,
# theta, about , -0.25, 0.05
# phi about -0.7 0.7


# the resolution of the LiDAR is 0.09 dgree for 5Hz. At 10Hz, the resolution is around 0.1728 degree.
# Ideally, W comes out to be 520

class DepImage:
    def __init__(self, feature, depth, thickness=1):
        self.feature = feature
        self.depth = depth
        self.thickness = thickness

def save_depth_jpg(depth, name:str):
    """
    save depth map to image file
    depth : tensor of shape (H, W)
    """
    depth = depth.cpu().numpy()
    gray = np.interp(depth, (depth.min(), depth.max()), (50,255))
    H, W = depth.shape
    imageio.imwrite(fig_path + name, gray.astype(np.uint8))

def xyz2range(points):
    """ convert points to depth map
        devide evenly, not wokring very well
    """
    shape = points.shape
    print("shape = ", shape)
    x = points[:, 0]  # -71~73
    y = points[:, 1]  # -21~53
    z = points[:, 2]  # -5~2.6
    intensity = points[:, 3]  # 0~1
    # convert xyz to theta-phi-r
    x2y2 = np.square(x)  + np.square(y)
    distance = np.sqrt(x2y2 + np.square(z))
    distance[distance == 0] = 1e-6
    thetas = np.arcsin(z / distance)
    phis = np.arcsin(-y / np.sqrt(x2y2))
    # print("min thetas = ", np.min(thetas))
    # print("max theats = ", np.max(thetas))
    # print("min phis = ", np.min(phis))
    # print("max phis = ", np.max(phis))
    # delta_phi = np.radians(90./512.)
    delta_phi = (np.max(phis) - np.min(phis)) / 511.
    # veldyne , resolution 26.8  vertical,
    # delta_theta = np.radians(26.8/64)
    delta_theta = (np.max(thetas) - np.min(thetas)) / 64.
    theta_idx = ((thetas - np.min(thetas)) / delta_theta).astype(int)
    phi_idx = ((phis - np.min(phis)) / delta_phi).astype(int)
    print("theta_idx = ", theta_idx)
    print("phi_idx = ", phi_idx)
    reflect_gray = (intensity * 255).astype(int)
    print("reflect_gray = ", reflect_gray)
    distance_gray = np.interp(distance, (distance.min(), distance.max()),
                             (50,255))
    H = 64
    W = 512
    C = 1
    range_map = np.zeros((H, W, C))
    range_map[63 - theta_idx,phi_idx,0] = distance_gray
    return range_map

def xyz2range_v2(points, ith=0, plot_diff=False, visualize=False):
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
    phi = np.arctan2(-y, x)
    # phi_p = np.arcsin(-y / np.sqrt(x ** 2 + y ** 2 ))
    # print("x rad diff", phi, phi_p, np.sum(phi - phi_p))
    angle_diff = np.diff(phi)
    if plot_diff:
        plt.plot(np.diff(phi))
        plt.savefig(fig_path+"angle_diff_{:d}.jpg".format(ith), dpi=300)
        plt.clf()
    threshold_angle = np.radians(10)  # huristic
    angle_diff = np.hstack((angle_diff, 0.0001)) # append one
    # new row when diff bigger than threashold
    angle_diff_mask = angle_diff > threshold_angle
    theta_idx = np.cumsum(angle_diff_mask)
    theta_idx[theta_idx >= 64] = 63
    # print("theta max min", theta_idx.max(), theta_idx.min())
    delta_phi = np.radians(90./512.)
    # delta_phi = np.radians((x.max() - x.min()) / 500.) # doenst work. many missing dots horiizontally
    phi_idx = np.floor((phi / delta_phi)).astype(int)
    phi_idx -= np.min(phi_idx) # translate to positive
    phi_idx[phi_idx >= 512] = 511
    # x_max = int(360.0 / h_res) #+ 1  # 投影后图片的宽度
    x_max = 512
    if visualize:
        depth_gray = np.interp(distance, (distance.min(), distance.max()), (20,255))
        depth_image = np.zeros((64, 512, 1))
        depth_image[theta_idx, phi_idx, 0] = depth_gray
        imageio.imwrite(fig_path + 'range_map_v2_{:d}.jpg'.format(ith),
            depth_image.astype(np.uint8))
    # 可能有 data loss， 有些数据点被覆盖了。
    depth_map = np.zeros((5, 64, 512), dtype=float) #+255
    depth_map[0, theta_idx, phi_idx] = x
    depth_map[1, theta_idx, phi_idx] = y
    depth_map[2, theta_idx, phi_idx] = z
    depth_map[3, theta_idx, phi_idx] = distance
    depth_map[4, theta_idx, phi_idx] = intensity
    return depth_map

def test_xyz2range_v2():
    kitti_info_path= "/home/gx/GitHub/second.pytorch/dataset/kitti_infos_val.pkl"
    kitti_root_path= "/home/gx/GitHub/second.pytorch/dataset/"
    fig_path = "output/"

    dataset = KittiDataset(kitti_root_path, kitti_info_path)
    for i in range(0,len(dataset)) :
        # break
        if i == 20:
            break
        item = dataset.get_sensor_data(i+8)
        # print(input_dict)
        points = item['lidar']['points']
        range_map = xyz2range(points)
        imageio.imwrite(fig_path + 'range_map_{:d}.jpg'.format(i),
            range_map.astype(np.uint8))
        range_map = xyz2range_v2(points, i, visualize=True)
        print("range map shape, ", range_map.shape)
        # 换个方式，根据 np.diff  把点填到不同的行。
        # 或者，用 max pooling ?
        # if i == 2:
        #     # plt.plot(ys, zs) #     fig, ax = plt.subplots() #     NPOINTS = len(ys)
        #     MAP='viridis'
        #     cm = plt.get_cmap(MAP)
        #     ax.set_prop_cycle(color=[cm(1.*i/(NPOINTS-1)) for i in range(NPOINTS-1)])
        #     for i in range(NPOINTS-1):
        #         ax.plot(ys[i:i+2],zs[i:i+2], '.')
        #     plt.xlabel('y')
        #     plt.ylabel('z')
        #     plt.savefig('yz.png')
        if False:
            # plot all dots
            fig, ax = plt.subplots()
            NPOINTS = len(thetas)
            MAP='viridis'
            cm = plt.get_cmap(MAP)
            frac = 1./(NPOINTS-1)
            ax.set_prop_cycle(color=[cm(i*frac) for i in range(NPOINTS-1)])
            for k in range(NPOINTS-1):
                ax.plot(phis[k:k+2],thetas[k:k+2], '.', markersize=1)
            plt.xlabel('phi')
            plt.ylabel('theta')
            plt.savefig(fig_path + 'phi-theta-{:d}.jpg'.format(i), dpi=300)
        # if i == 2:
        #     BINS= 4000
        #     fig, ax = plt.subplots()
        #     ax.hist(thetas, BINS,cumulative=True )
        #     ax.set_xlabel("points")
        #     ax.set_ylabel("thetas accumulate")
        #     plt.savefig(fig_path + 'thetas-hist.jpg', dpi=300)
        #     fig, ax = plt.subplots()
        #     ax.hist(phis, BINS,cumulative=True )
        #     ax.set_xlabel("points")
        #     ax.set_ylabel("phis accumulate")
        #     plt.savefig(fig_path + 'phis-hist.jpg', dpi=300)
                # in_channels,
                # out_channels,
                # kernel_size,
                # stride=1,
                # paddding=0,
                # dilatoin=1,
                # groups=1,
        # Depth convolution

def _calculate_fan_in_and_fan_out_hwio(tensor):
    dimensions = tensor.ndimension()
    if dimensions < 2:
        raise ValueError(
            "Fan in and fan out can not be computed for tensor with fewer than 2 dimensions"
        )

    if dimensions == 2:  # Linear
        fan_in = tensor.size(-2)
        fan_out = tensor.size(-1)
    else:
        num_input_fmaps = tensor.size(-2)
        num_output_fmaps = tensor.size(-1)
        receptive_field_size = 1
        if tensor.dim() > 2:
            receptive_field_size = tensor[..., 0, 0].numel()
        fan_in = num_input_fmaps * receptive_field_size
        fan_out = num_output_fmaps * receptive_field_size

    return fan_in, fan_out

class DepConv3D(nn.Module):
    def __init__(self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=False,
        subm=False):
        super(DepConv3D, self).__init__()

        if not isinstance(kernel_size, (list, tuple)):
            kernel_size = [kernel_size] * 3
        assert groups == 1, "groups not supported yet"

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        # self.conv1x1
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        # self.bias = bias
        self.subm = subm

        self.weight = nn.Parameter(
            torch.Tensor(out_channels, in_channels, *kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            # self.register_parameter('bias', None)
            self.bias = torch.empty(0)
        self.reset_parameters()

    def reset_parameters(self):
        n = self.in_channels
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = _calculate_fan_in_and_fan_out_hwio(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, features, depth):
        out_tensor = depconv3d(features, depth, self.weight,
            self.bias, self.stride, self.padding, self.dilation,
            self.groups)

        return out_tensor

def init_depth_from_feature(feature, k):
    """
        return a depth tensor of shape (B, H, W)
        max depth be 'k'

    Argument:
        feature: tensor of shape(B, C, H, W),
            C = 5
    """
    distance = feature[:, 3, :, :]
    depth = ((distance - distance.min()) * k / (distance.max()- distance.min())).long()
    return depth

class ConvNet(nn.Module):
    def __init__(self,
        num_input_features,
        use_norm=True):
        super(ConvNet, self).__init__()
        if use_norm:
            BatchNorm2d = change_default_args(
                eps=1e-3, momentum=0.01)(nn.BatchNorm2d)
            BatchNorm1d = change_default_args(
                eps=1e-3, momentum=0.01)(nn.BatchNorm1d)
            Conv2d = change_default_args(bias=False)(nn.Conv2d)
        else:
            BatchNorm2d = Empty
            BatchNorm1d = Empty
            Conv2d = change_default_args(bias=True)(nn.Conv2d)

        # 128 * 64 * 512
        self.conv1 = nn.Conv2d(num_input_features, 8, 3, padding=1)
        self.bn1 = BatchNorm2d(8)
        self.conv2 = nn.Conv2d(8, 8, 3, padding=1)
        self.bn2 = BatchNorm2d(8)
        self.conv3 = nn.Conv2d(8, 16, 3, (1,2), padding=1) # 128 * 64 * 256
        self.bn3 = BatchNorm2d(16)
        self.conv4 = nn.Conv2d(16, 16, 3, padding=1)
        self.bn4 = BatchNorm2d(16)
        self.conv5 = nn.Conv2d(16, 32, 3, 2, padding=1) # 64,32,128
        self.bn5 = BatchNorm2d(32)
        # self.conv6 = nn.Conv2d(32, 32, 3, padding=1)
        # self.bn6 = BatchNorm2d(32)
        # self.conv7 = nn.Conv2d(32, 32, 3, padding=1)
        # self.bn7 = BatchNorm2d(32)
        self.conv8 = nn.Conv2d(32, 32, 3, padding=1)
        self.bn8 = BatchNorm2d(32)
        self.conv9 = nn.Conv2d(32, 64, 3, (1,2), padding=1) # 32, 32 * 64
        self.bn9 = BatchNorm2d(64)
        # self.conv10 = nn.Conv2d(32, 32, 3,padding=1)
        # self.bn10 = BatchNorm2d(32)
        self.conv11 = nn.Conv2d(64, 64, 3, (1,1), padding=1)# 16
        self.bn11 = BatchNorm2d(64)
        self.conv12 = nn.Conv2d(64, 128, 3, (1,1), padding=1) # 8 * 32 * 64
        self.bn12 = BatchNorm2d(128)
        self.conv13 = nn.Conv2d(128, 256, 3, (1,1), padding=1) # 4
        self.bn13 = BatchNorm2d(256)
        # self.conv14 = nn.Conv2d(32, 32, 3, (2,1,1), padding=1) # 2
        # self.bn14 = BatchNorm2d(32)

        self.count = 0

    def forward(self, feature):

        depth = init_depth_from_feature(feature, 128)
        # with torch.no_grad():
        #     print("depth.shape, depth.shape")
        #     print("depth,", depth[0])
        #     save_depth_jpg(depth[0], "{:2d}.jpg".format(self.count))
        # self.count += 1

        # print("depth =", depth)
        # print("depth min, max, shape", depth.min(), depth.max(), depth.shape)
        # print("feature shape ", feature.shape)
        # print("depth =", depth)
        # print("depth min, max, shape", depth.min(), depth.max(), depth.shape)
        # print("feature shape ", feature.shape)

        x = self.conv1(feature)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.conv3(x)
        # depth = F.max_pool2d(depth.float(), 3, padding=1, stride=(1,2)).long()
        x = self.bn3(x)
        x = F.relu(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = F.relu(x)
        x = self.conv5(x)
        # depth = F.max_pool2d(depth.float(), 3, padding=1, stride=(2,2)).long()
        # depth = depth // 2
        x = self.bn5(x)
        x = F.relu(x)
        # x = self.conv6(x)
        # x = self.bn6(x)
        # x = F.relu(x)
        # x = self.conv7(x)
        # x = self.bn7(x)
        # x = F.relu(x)
        x = self.conv8(x)
        x = self.bn8(x)
        x = F.relu(x)
        x = self.conv9(x)
        # depth = F.max_pool2d(depth.float(), 3, padding=1, stride=(1,2)).long()
        # depth = depth // 2
        x = self.bn9(x)
        x = F.relu(x)
        # x = self.conv10(x)
        # x = self.bn10(x)
        # x = F.relu(x)
        x = self.conv11(x)
        # depth = depth // 2
        x = self.bn11(x)
        x = F.relu(x)
        x = self.conv12(x)
        # depth = depth // 2
        x = self.bn12(x)
        x = F.relu(x)
        x = self.conv13(x)
        # depth = depth // 2
        x = self.bn13(x)
        x = F.relu(x)
        # x = self.conv14(x)
        # depth = depth // 2
        # x = self.bn14(x)
        # x = F.relu(x)

        return x


class DepConvNet(nn.Module):
    def __init__(self,
        num_input_features,
        use_norm=True):
        super(DepConvNet, self).__init__()
        if use_norm:
            BatchNorm2d = change_default_args(
                eps=1e-3, momentum=0.01)(nn.BatchNorm2d)
            BatchNorm1d = change_default_args(
                eps=1e-3, momentum=0.01)(nn.BatchNorm1d)
            Conv2d = change_default_args(bias=False)(nn.Conv2d)
        else:
            BatchNorm2d = Empty
            BatchNorm1d = Empty
            Conv2d = change_default_args(bias=True)(nn.Conv2d)

        # 128 * 64 * 512
        self.conv1 = DepConv3D(num_input_features, 8, 3, padding=1)
        self.bn1 = BatchNorm2d(8)
        self.conv2 = DepConv3D(8, 8, 3, padding=1)
        self.bn2 = BatchNorm2d(8)
        self.conv3 = DepConv3D(8, 16, 3, (1,1,2), padding=1) # 128 * 64 * 256
        self.bn3 = BatchNorm2d(16)
        # self.conv4 = DepConv3D(16, 16, 3, padding=1)
        # self.bn4 = BatchNorm2d(16)
        self.conv5 = DepConv3D(16, 32, 3, 2, padding=1) # 64,32,128
        self.bn5 = BatchNorm2d(32)
        # self.conv6 = DepConv3D(32, 32, 3, padding=1)
        # self.bn6 = BatchNorm2d(32)
        # self.conv7 = DepConv3D(32, 32, 3, padding=1)
        # self.bn7 = BatchNorm2d(32)
        # self.conv8 = DepConv3D(32, 32, 3, padding=1)
        # self.bn8 = BatchNorm2d(32)
        self.conv9 = DepConv3D(32, 32, 3, (2,1,2), padding=1) # 32, 32 * 64
        self.bn9 = BatchNorm2d(32)
        # self.conv10 = DepConv3D(32, 32, 3,padding=1)
        # self.bn10 = BatchNorm2d(32)
        self.conv11 = DepConv3D(32, 32, 3, (2,1,1), padding=1)# 16
        self.bn11 = BatchNorm2d(32)
        self.conv12 = DepConv3D(32, 32, 3, (2,1,1), padding=1) # 8 * 32 * 64
        self.bn12 = BatchNorm2d(32)
        self.conv13 = DepConv3D(32, 32, 3, (2,1,1), padding=1) # 4
        self.bn13 = BatchNorm2d(32)
        # self.conv14 = DepConv3D(32, 32, 3, (2,1,1), padding=1) # 2
        # self.bn14 = BatchNorm2d(32)

        self.count = 0

    def forward(self, feature):

        depth = init_depth_from_feature(feature, 128)
        # with torch.no_grad():
        #     print("depth.shape, depth.shape")
        #     print("depth,", depth[0])
        #     save_depth_jpg(depth[0], "{:2d}.jpg".format(self.count))
        # self.count += 1

        # print("depth =", depth)
        # print("depth min, max, shape", depth.min(), depth.max(), depth.shape)
        # print("feature shape ", feature.shape)
        # print("depth =", depth)
        # print("depth min, max, shape", depth.min(), depth.max(), depth.shape)
        # print("feature shape ", feature.shape)

        x = self.conv1(feature, depth)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.conv2(x, depth)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.conv3(x, depth)
        depth = F.max_pool2d(depth.float(), 3, padding=1, stride=(1,2)).long()
        x = self.bn3(x)
        x = F.relu(x)
        # x = self.conv4(x, depth)
        # x = self.bn4(x)
        # x = F.relu(x)
        x = self.conv5(x, depth)
        depth = F.max_pool2d(depth.float(), 3, padding=1, stride=(2,2)).long()
        depth = depth // 2
        x = self.bn5(x)
        x = F.relu(x)
        # x = self.conv6(x, depth)
        # x = self.bn6(x)
        # x = F.relu(x)
        # x = self.conv7(x, depth)
        # x = self.bn7(x)
        # x = F.relu(x)
        # x = self.conv8(x, depth)
        # x = self.bn8(x)
        # x = F.relu(x)
        x = self.conv9(x, depth)
        depth = F.max_pool2d(depth.float(), 3, padding=1, stride=(1,2)).long()
        depth = depth // 2
        x = self.bn9(x)
        x = F.relu(x)
        # x = self.conv10(x, depth)
        # x = self.bn10(x)
        # x = F.relu(x)
        x = self.conv11(x, depth)
        depth = depth // 2
        x = self.bn11(x)
        x = F.relu(x)
        x = self.conv12(x, depth)
        depth = depth // 2
        x = self.bn12(x)
        x = F.relu(x)
        x = self.conv13(x, depth)
        depth = depth // 2
        x = self.bn13(x)
        x = F.relu(x)
        # x = self.conv14(x, depth)
        # depth = depth // 2
        # x = self.bn14(x)
        # x = F.relu(x)

        return x


class DepConvNet2(nn.Module):
    def __init__(self,
        num_input_features,
        use_norm=True):
        super(DepConvNet2, self).__init__()
        if use_norm:
            BatchNorm2d = change_default_args(
                eps=1e-3, momentum=0.01)(nn.BatchNorm2d)
            BatchNorm1d = change_default_args(
                eps=1e-3, momentum=0.01)(nn.BatchNorm1d)
            Conv2d = change_default_args(bias=False)(nn.Conv2d)
        else:
            BatchNorm2d = Empty
            BatchNorm1d = Empty
            Conv2d = change_default_args(bias=True)(nn.Conv2d)

        # 512 * 64 * 512
        self.conv1 = DepConv3D(num_input_features, 8, 3, padding=1)
        self.bn1 = BatchNorm2d(8)
        self.conv2 = DepConv3D(8, 8, 3, padding=1)
        self.bn2 = BatchNorm2d(8)
        self.conv3 = DepConv3D(8, 16, 3, (1,1,2), padding=1) # 512 * 64 * 256
        self.bn3 = BatchNorm2d(16)
        self.conv4 = DepConv3D(16, 16, 3, padding=1)
        self.bn4 = BatchNorm2d(16)
        self.conv16 = DepConv3D(16, 16, 3, padding=1)
        self.bn16 = BatchNorm2d(16)
        self.conv5 = DepConv3D(16, 32, 3, 2, padding=1) # 256 * 32 * 128
        self.bn5 = BatchNorm2d(32)
        # self.conv6 = DepConv3D(32, 32, 3, padding=1)
        # self.bn6 = BatchNorm2d(32)
        # self.conv7 = DepConv3D(32, 32, 3, padding=1)
        # self.bn7 = BatchNorm2d(32)
        # self.conv8 = DepConv3D(32, 32, 3, padding=1)
        # self.bn8 = BatchNorm2d(32)
        self.conv9 = DepConv3D(32, 32, 3, (2,1,2), padding=1) # 128, 32, 64
        self.bn9 = BatchNorm2d(32)
        self.conv10 = DepConv3D(32, 32, 3, padding=1)
        self.bn10 = BatchNorm2d(32)
        self.conv11 = DepConv3D(32, 32, 3, (2,2,1), padding=1)# 64, 16, 64
        self.bn11 = BatchNorm2d(32)
        # self.conv12 = DepConv3D(32, 32, 3, (2,1,1), padding=1) # 8 * 32 * 64
        # self.bn12 = BatchNorm2d(32)
        # self.conv13 = DepConv3D(32, 32, 3, (2,1,1), padding=1) # 4
        # self.bn13 = BatchNorm2d(32)
        # self.conv14 = DepConv3D(32, 32, 3, (2,1,1), padding=1) # 2
        # self.bn14 = BatchNorm2d(32)

        self.count = 0

    def forward(self, feature):

        depth = init_depth_from_feature(feature, 512)
        # with torch.no_grad():
        #     print("depth.shape, depth.shape")
        #     print("depth,", depth[0])
        #     save_depth_jpg(depth[0], "{:2d}.jpg".format(self.count))
        # self.count += 1

        # print("depth =", depth)
        # print("depth min, max, shape", depth.min(), depth.max(), depth.shape)
        # print("feature shape ", feature.shape)
        # print("depth =", depth)
        # print("depth min, max, shape", depth.min(), depth.max(), depth.shape)
        # print("feature shape ", feature.shape)
        x = self.conv1(feature, depth)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.conv2(x, depth)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.conv3(x, depth)
        depth = F.max_pool2d(depth.float(), 3, padding=1, stride=(1,2)).long()
        x = self.bn3(x)
        x = F.relu(x)
        x = self.conv4(x, depth)
        x = self.bn4(x)
        x = F.relu(x)
        x = self.conv16(x, depth)
        x = self.bn16(x)
        x = F.relu(x)
        x = self.conv5(x, depth)
        depth = F.max_pool2d(depth.float(), 3, padding=1, stride=(2,2)).long()
        depth = depth // 2
        x = self.bn5(x)
        x = F.relu(x)
        # x = self.conv6(x, depth)
        # x = self.bn6(x)
        # x = F.relu(x)
        # x = self.conv7(x, depth)
        # x = self.bn7(x)
        # x = F.relu(x)
        # x = self.conv8(x, depth)
        # x = self.bn8(x)
        # x = F.relu(x)
        x = self.conv9(x, depth)
        depth = F.max_pool2d(depth.float(), 3, padding=1, stride=(1,2)).long()
        depth = depth // 2
        x = self.bn9(x)
        x = F.relu(x)
        # x = self.conv10(x, depth)
        # x = self.bn10(x)
        # x = F.relu(x)
        x = self.conv11(x, depth)
        depth = F.max_pool2d(depth.float(), 3, padding=1, stride=(2,1)).long()
        depth = depth // 2
        x = self.bn11(x)
        x = F.relu(x)
        # x = self.conv12(x, depth)
        # depth = depth // 2
        # x = self.bn12(x)
        # x = F.relu(x)
        # x = self.conv13(x, depth)
        # depth = depth // 2
        # x = self.bn13(x)
        # x = F.relu(x)
        # x = self.conv14(x, depth)
        # depth = depth // 2
        # x = self.bn14(x)
        # x = F.relu(x)
        x = depth_to_3D(x, depth, 64)
        B, C, D, H, W = x.shape
        # print("shape , ", B,C,D,H,W)
        return x.permute(0,1,3,2,4).reshape(B, C*H, D, W)

def depconv3d(input, depth, weight,
        bias=torch.tensor([]),
        stride=1,
        padding=0,
        dilation=1,
        groups=1
        ):
    """
    Applies a 3D convolution on 2D input with depth info.

    Parameters:

        input: the input tensor of shape (minibatch, in_channels, iH, iW)
        depth: depth tensor of shape (minibatch, iH, iW), of type int
        weight: a 3d filter of shape
                    (out_channels, in_channels/groups, kD, kH, kW)
        bias: optional bias tensor of shape (out_channels). Default: None
        stride: the stride of the cconvolving kernel, can be a single number or
            a tuple (sD, sH, sW). Default: 1
        padding: implicit paddings on both sides of the input. Can be a single
            number or a tuple (padH, padW). Default: 0
        dilation: the spacing between kernel elements. Can be a snigle number
            or a tuple (dD, dH, dW). Default: 1
        groups: split into groups, in_channels shouldd be divisible by the
            number of groups. Default: 1

        see https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md

    Returns:
        a tensor of shape (N, out_channels, oH, oW), where

            oH = floor((iH + 2 x padH - dH x (kH - 1) -1) / sH + 1)
            oW = floor((iW + 2 x padW - dW x (kW - 1) -1) / sW + 1)

    Examples:
        filters = torch.torch.randn(33, 16, 3, 3)
        depth = torch.randn()
    """
    # group not supported yet
    assert groups == 1
    assert input.dim() == 4
    assert depth.dim() == 3
    assert weight.dim() == 5

    if not isinstance(stride, (list, tuple)):
        stride = [stride] * 3
    if not isinstance(padding, (list, tuple)):
        padding = [padding] * 3
    if not isinstance(dilation, (list, tuple)):
        dilation = [dilation] * 3

    return depconv3d_(input, depth, weight,
        bias, stride, padding, dilation, groups)

@torch.jit.script
def depconv3d_(input, depth, weight,
        bias:torch.Tensor,
        stride:List[int],
        padding:List[int],
        dilation:List[int],
        groups:int):

    B, iC, iH, iW = input.shape
    oC, iC, kD, kH, kW = weight.shape
    sD, sH, sW = stride
    padD, padH, padW = padding
    dD, dH, dW = dilation
    assert (kD == 3), "only support weight of shape 3x3x3 now"
    assert (dD == 1 and dH == 1 and dW == 1), "only support dilation of 1x1x1"

    # print("input dtype", input.dtype)
    # im2colo
    # unfoldd
    unfolded_input = F.unfold(input, kernel_size=(kH, kW), dilation=dilation[1:], padding=padding[1:], stride=stride[1:])
        # B, iC * kH * kW, N
    # unfold does not support longTensor
    depth_ = depth.unsqueeze(1) # B, 1, iH, iW
    unfolded_depth = F.unfold(depth_.float(), kernel_size=(kH, kW),
         dilation=dilation[1:], padding=padding[1:], stride=stride[1:]).long()
         # B , kH * kW * 1 , N
    N = unfolded_depth.size(-1)

    # extend weight with zeros, for indexing out of range values
    ext_weight = torch.cat((weight, torch.zeros([oC, iC, 1, kH, kW], device=torch.device('cuda'))), dim=2)
        # oC, iC, kD+1, kH, kW

    # depth difference
    depth_diff = (unfolded_depth - unfolded_depth[:, kH * kW // 2, :].unsqueeze(1))
    # B, kH * kW * 1, N

    # in range
    mask_diff_low = depth_diff.lt(-1)
    mask_diff_high = depth_diff.ge(1)
    depth_diff[mask_diff_low] = torch.tensor(2)
    depth_diff[mask_diff_high] = torch.tensor(2)
    depth_index = depth_diff.add(1).unsqueeze(1) # B, kH*kW*1 , N

    depth_index = depth_index.reshape(B, 1, 1, 1, kH *kW, N).expand(-1, oC, iC, -1, -1, -1) # B, oC, iC, 1, kH*kW, N

    b_weight = ext_weight.reshape(oC, iC, kD + 1, kH * kW).unsqueeze(0) \
        .unsqueeze(-1).expand(B, -1, -1, -1, -1, N)  # B, oC, iC, kD, kH*kW, N

    unfolded_weight = b_weight.gather(3, depth_index).reshape(B, oC, iC * kH * kW, N) # B, oC, iC * kH * kW, N
    # print("type",(unfolded_weight.dtype), (unfolded_input.dtype))
    # print("enisum shape ", unfolded_weight.shape, unfolded_input.shape)
    unfolded_output = torch.einsum('bikj,bkj->bij', unfolded_weight, unfolded_input)
    # B, oC, N
    oH = int(torch.floor((iH + 2 * padH - dH * (kH - 1) -1) / sH + 1))
    oW = int(torch.floor((iW + 2 * padW - dW * (kW - 1) -1) / sW + 1))
    #
    output = F.fold(unfolded_output, output_size=(oH, oW), kernel_size=(1,1)) # B, oC, oH, oW

    # change the depth
    # depth = ((depth + padD - dD * (kD // 2)) // sD + 1).long()
    # depth

    # if bias is not None:
    #     output += bias.unsqueeze(0).expand_as(output)
    # output = output.reshape()
    return output




def depth_to_3D(feature, depth, D=0):
    """
    convert (depth, feature) to a 3D tensor

    Aruguments:
        feature : tensor of shape (batchsize, Channel, H, W)
        depth : long tensor of shape (batchsize, H, W)

    Returns:
        a 3D tensor of size (batchsize, Channel, D, H, W)
    """
    # if not isinstance(depth, torch.LongTensor):
    #      raise ValueError("'depth' should be of type LongTensor,"
    #         " but got %s" % type(depth))

    if D == 0:
        D_min, D_max = depth.min(), depth.max()
        D = D_max - D_min + 1
    B, C, H, W = feature.shape
    device = feature.get_device()
    ret_tensor = torch.zeros(B, C, D, H, W, device=device)
    feature_ = feature.reshape(B, C, 1, H, W).expand_as(ret_tensor)
    depth_idx = depth.reshape(B, 1, 1, H, W).expand_as(ret_tensor)
    depth_idx[depth_idx >= D]  = D - 1
    # expand to shape, B, C, D, H, W
    ret_tensor.scatter_(2, depth_idx, feature_)
    return ret_tensor

def _test_depth_to_3D():

    feature_map = torch.tensor(
        [[.1,.2,.3],
        [.4,.5,.6],
        [.7,.8,.9]]).reshape(1,1,3,3).cuda()

    depth_map = torch.tensor(
        [[0,0,1],
         [1,1,0],
          [1,1,1]]).reshape(1,3,3).cuda()

    v3d = depth_to_3D(feature_map, depth_map)

    v3d_expected = torch.tensor([
        [[0.1000, 0.2000, 0.0000],
           [0.0000, 0.0000, 0.6000],
           [0.0000, 0.0000, 0.0000]],
          [[0.0000, 0.0000, 0.3000],
           [0.4000, 0.5000, 0.0000],
           [0.7000, 0.8000, 0.9000]]
        ]).reshape(1,1,2,3,3).cuda()

    print(torch.allclose(v3d, v3d_expected))

def test_submanifold_conv3d():
    """ compare
    compare the result of dense 3D conv and depConv
    """

    def _test(feature, depth, weight, stride=1):
        # submanifold
        res = depconv3d(feature, depth, weight, padding=1, stride=stride)
        print("res shape = ", res.shape)
        # print("depth ",  depth)
        # print(" res depth ", res_depth)

        # dense as reference
        feature_dense = depth_to_3D(feature, depth)
        res_ref = F.conv3d(feature_dense, weight, padding=1, stride=stride)
        print("res ref shape = ", res_ref.shape)

        res_dense = depth_to_3D(res, depth)

        # print("res")
        # print(res_dense)
        # print("ref")
        # print(res_ref)

        is_close = torch.allclose(res_ref[res_dense != 0], res_dense[res_dense != 0],
            rtol=1e-1)
        print("diff = ", torch.sum(torch.abs(res_ref[res_dense!=0] - res_dense[res_dense != 0])))
        print(is_close)

    feature = torch.tensor(
        [[.1,.2,.3],
        [.4,.5,.6],
        [.7,.8,.9]]).reshape(1,1,3,3).cuda()

    depth = torch.tensor(
        [[0,0,1],
         [1,1,0],
         [1,1,2]]).reshape(1,3,3).cuda()
    weight = torch.randn(1,1,3,3,3).cuda()
    _test(feature, depth, weight)

    def rand_test(B, iC, oC, H, W, k=3, stride=1):
        feature = torch.randn(B, iC, H, W).cuda()
        depth = (torch.randn(B, H, W)).long()
        depth = (depth - depth.min()).cuda()
        print("depth max =", depth.max().cpu())
        if not isinstance(k, (list, tuple)):
            k = [k] * 3
        weight = torch.randn(oC, iC, *k).cuda()
        _test(feature, depth, weight, stride)

    rand_test(2, 3, 3, 8, 8)
    # rand_test(2, 3, 3, 8, 8, stride=(2,1,2))
    # rand_test(1, 3, 3, 8, 8, stride=2)
    rand_test(2, 16, 16, 128, 128)



def testDConv():
    """
    the test use kitti dataset,
    and compares the result of DepConv with torch's 3d conv
    """
    shapes[[19,18,17]]
    batchsizes = [1,2]

    in_channels = [64]
    out_channels = [32]
    ksizes = [3]
    # strides = [1, 2, 3]
    strides=[1]
    # paddings = [0, 1, 2]
    paddings = [0]
    # dilations = [1, 2, 3]
    dilations = [1]

    for shape, bs, iC, oC, k, s  in params_grid(shapes, batchsizes, in_channels,
        out_channels, ksizes, strides):
        pass

# _test_depth_to_3D()

# test_submanifold_conv3d()

# test_xyz2range_v2()
