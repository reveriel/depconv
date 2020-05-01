
from second.data.kitti_dataset import KittiDataset
import matplotlib.pyplot as plt
import matplotlib.collections as mcoll
import numpy as np
import math
import imageio
import torch
import torch.nn.functional as F
from torch import nn
from typing import Dict, List, Tuple, Union

# from test_utils import params_grid, generate_sparse_data, TestCase
import unittest
from torchplus.tools import change_default_args
import math

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
        print(item)
        points = item['lidar']['points']
        # range_map = xyz2range(points)
        # imageio.imwrite(fig_path + 'range_map_{:d}.jpg'.format(i),
            # range_map.astype(np.uint8))
        # range_map = xyz2range_v2(points, i, visualize=True)
        # print("range map shape, ", range_map.shape)
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

test_xyz2range_v2()