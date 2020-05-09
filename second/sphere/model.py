
import torch
import torch.nn.functional as F
from torch import nn
from torchplus.tools import change_default_args
from second.sphere.conv import (DepConv3D, DepConv3D_v2, init_depth_from_feature,
   depth_to_3D, depth_to_3D_v2)


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
            BatchNorm2d = nn.Empty
            BatchNorm1d = nn.Empty
            Conv2d = change_default_args(bias=True)(nn.Conv2d)

        # 512 * 64 * 512
        self.conv1 = DepConv3D(num_input_features, 16, 3, padding=1)
        self.bn1 = BatchNorm2d(16)
        self.conv2 = DepConv3D(16, 16, 3, padding=1)
        self.bn2 = BatchNorm2d(16)
        # self.conv3 = DepConv3D(16, 16, 3, padding=1) # 512 * 64 * 256
        # self.bn3 = BatchNorm2d(16)
        # self.conv4 = DepConv3D(16, 16, 3, padding=1)
        # self.bn4 = BatchNorm2d(16)
        # self.conv16 = DepConv3D(16, 16, 3, padding=1)
        # self.bn16 = BatchNorm2d(16)
        self.conv5 = DepConv3D(16, 32, 3, 2, padding=1) # 256 * 32 * 128
        self.bn5 = BatchNorm2d(32)
        self.conv6 = DepConv3D(32, 32, 3, padding=1)
        self.bn6 = BatchNorm2d(32)
        self.conv7 = DepConv3D(32, 32, 3, padding=1)
        self.bn7 = BatchNorm2d(32)
        # self.conv8 = DepConv3D(32, 32, 3, padding=1)
        # self.bn8 = BatchNorm2d(32)
        self.conv9 = DepConv3D(32, 64, 3, (2,2,2), padding=1) # 128, 32, 64
        self.bn9 = BatchNorm2d(64)
        self.conv20 = DepConv3D(64, 64, 3, padding=1)
        self.bn20 = BatchNorm2d(64)
        self.conv21 = DepConv3D(64, 64, 3, padding=1)
        self.bn21 = BatchNorm2d(64)
        self.conv22 = DepConv3D(64, 64, 3, padding=1)
        self.bn22 = BatchNorm2d(64)
        self.conv11 = DepConv3D(64, 64, 3, (2,2,2), padding=1)# 64, 16, 64
        self.bn11 = BatchNorm2d(64)
        self.conv12 = DepConv3D(64, 64, 3, padding=1) # 8 * 32 * 64
        self.bn12 = BatchNorm2d(64)
        self.conv13 = DepConv3D(64, 64, 3, padding=1) # 4
        self.bn13 = BatchNorm2d(64)
        # self.conv14 = DepConv3D(64, 64, 3, padding=1) # 2
        # self.bn14 = BatchNorm2d(64)
        self.conv15 = DepConv3D(64, 64, 3, (1,2,1), padding=1)
        self.bn15 = BatchNorm2d(64)
        self.conv16 = DepConv3D(64, 64, 3, (1,2,1), padding=1)
        self.bn16 = BatchNorm2d(64)
        # self.conv17 = DepConv3D(64, 64, 3, (1,2,1), padding=1)
        # self.bn17 = BatchNorm2d(64)

        self.count = 0

    def forward(self, feature):

        with torch.no_grad():
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
        # depthmap_bev(x, depth)

        x = self.conv2(x, depth)
        x = self.bn2(x)
        x = F.relu(x)

        # depthmap_bev(x, depth)
        # x = self.conv3(x, depth)
        # depth = F.max_pool2d(depth.float(), 3, padding=1, stride=(1,2)).long()
        # x = self.bn3(x)
        # x = F.relu(x)
        # x = self.conv4(x, depth)
        # x = self.bn4(x)
        # x = F.relu(x)
        # x = self.conv16(x, depth)
        # x = self.bn16(x)
        # x = F.relu(x)
        x = self.conv5(x, depth)
        depth = F.max_pool2d(depth.float(), 3, padding=1, stride=(2,2)).long()
        depth = depth // 2
        x = self.bn5(x)
        x = F.relu(x)
        x = self.conv6(x, depth)
        x = self.bn6(x)
        x = F.relu(x)
        x = self.conv7(x, depth)
        x = self.bn7(x)
        x = F.relu(x)


        # x = self.conv8(x, depth)
        # x = self.bn8(x)
        # x = F.relu(x)
        x = self.conv9(x, depth)
        depth = F.max_pool2d(depth.float(), 3, padding=1, stride=(2,2)).long()
        depth = depth // 2
        x = self.bn9(x)
        x = F.relu(x)
        x = self.conv20(x, depth)
        x = self.bn20(x)
        x = F.relu(x)
        # with torch.no_grad():
        #     depthmap_bev(x, depth)
        x = self.conv21(x, depth)
        x = self.bn21(x)
        x = F.relu(x)
        x = self.conv22(x, depth)
        x = self.bn22(x)
        x = F.relu(x)
        x = self.conv11(x, depth)
        depth = F.max_pool2d(depth.float(), 3, padding=1, stride=(2,2)).long()
        depth = depth // 2
        x = self.bn11(x)
        x = F.relu(x)
        x = self.conv12(x, depth)
        x = self.bn12(x)
        x = F.relu(x)
        x = self.conv13(x, depth)
        x = self.bn13(x)
        x = F.relu(x)
        # x = self.conv14(x, depth)
        # depth = depth // 2
        # x = self.bn14(x)
        # x = F.relu(x)
        x = self.conv15(x, depth)
        depth = F.max_pool2d(depth.float(), 3, padding=1, stride=(2,1)).long()
        x = self.bn15(x)
        x = F.relu(x)
        x = self.conv16(x, depth)
        depth = F.max_pool2d(depth.float(), 3, padding=1, stride=(2,1)).long()
        x = self.bn16(x)
        x = F.relu(x)
        # x = self.conv17(x, depth)
        # depth = F.max_pool2d(depth.float(), 3, padding=1, stride=(2,1)).long()
        # x = self.bn17(x)
        # x = F.relu(x)

        x = depth_to_3D(x, depth, 64)
        B, C, D, H, W = x.shape
        # with torch.no_grad():
        #     # f =
        #     for c in range(C):
        #         save_2d_jpg(x[0][c].sum(dim=0), "bevmap-%d.jpg"%(c))

        print("shape BCDHW, ", B,C,D,H,W)
        return x.permute(0,1,3,2,4).reshape(B, C*H, D, W)



class DepConvNet3(nn.Module):
    def __init__(self,
        num_input_features,
        use_norm=True):
        super(DepConvNet3, self).__init__()
        if use_norm:
            BatchNorm2d = change_default_args(
                eps=1e-3, momentum=0.01)(nn.BatchNorm2d)
            BatchNorm1d = change_default_args(
                eps=1e-3, momentum=0.01)(nn.BatchNorm1d)
            Conv2d = change_default_args(bias=False)(nn.Conv2d)
        else:
            BatchNorm2d = nn.Empty
            BatchNorm1d = nn.Empty
            Conv2d = change_default_args(bias=True)(nn.Conv2d)

        # 512 * 64 * 512
        self.conv1 = DepConv3D(num_input_features, 16, 3, padding=1)
        self.bn1 = BatchNorm2d(16)
        self.conv2 = DepConv3D(16, 16, 3, padding=1)
        self.bn2 = BatchNorm2d(16)
        # self.conv3 = DepConv3D(16, 16, 3, padding=1) # 512 * 64 * 256
        # self.bn3 = BatchNorm2d(16)
        # self.conv4 = DepConv3D(16, 16, 3, padding=1)
        # self.bn4 = BatchNorm2d(16)
        # self.conv16 = DepConv3D(16, 16, 3, padding=1)
        # self.bn16 = BatchNorm2d(16)
        self.conv5 = DepConv3D_v2(16, 32, 3, 2, padding=1, subm=True) # 256 * 32 * 128
        self.bn5 = BatchNorm2d(32)
        self.conv6 = DepConv3D_v2(32, 32, 3, padding=1, subm=False)
        self.bn6 = BatchNorm2d(32)
        self.conv7 = DepConv3D_v2(32, 32, 3, padding=1, subm=False)
        self.bn7 = BatchNorm2d(32)
        # self.conv8 = DepConv3D(32, 32, 3, padding=1)
        # self.bn8 = BatchNorm2d(32)
        self.conv9 = DepConv3D_v2(32, 64, 3, (2,2,2), padding=1, subm=True) # 128, 32, 64
        self.bn9 = BatchNorm2d(64)
        self.conv20 = DepConv3D_v2(64, 64, 3, padding=1, subm=False)
        self.bn20 = BatchNorm2d(64)
        self.conv21 = DepConv3D_v2(64, 64, 3, padding=1, subm=False)
        self.bn21 = BatchNorm2d(64)
        self.conv22 = DepConv3D_v2(64, 64, 3, padding=1, subm=False)
        self.bn22 = BatchNorm2d(64)
        self.conv11 = DepConv3D_v2(64, 64, 3, (2,2,2), padding=1, subm=True)# 64, 16, 64
        self.bn11 = BatchNorm2d(64)
        self.conv12 = DepConv3D_v2(64, 64, 3, padding=1, subm=False) # 8 * 32 * 64
        self.bn12 = BatchNorm2d(64)
        self.conv13 = DepConv3D_v2(64, 64, 3, padding=1, subm=False) # 4
        self.bn13 = BatchNorm2d(64)
        # self.conv14 = DepConv3D(64, 64, 3, padding=1) # 2
        # self.bn14 = BatchNorm2d(64)
        self.conv15 = DepConv3D_v2(64, 64, 3, (1,2,1), padding=1,subm=False)
        self.bn15 = BatchNorm2d(64)
        self.conv16 = DepConv3D_v2(64, 64, 3, (1,2,1), padding=1,subm=False)
        self.bn16 = BatchNorm2d(64)
        # self.conv17 = DepConv3D(64, 64, 3, (1,2,1), padding=1)
        # self.bn17 = BatchNorm2d(64)

        self.count = 0

    def forward(self, feature):

        if feature.dim() == 3:
            feature = feature.unsqueeze(0)

        with torch.no_grad():
            depth = init_depth_from_feature(feature, 512)
        print("1 depth max min=", depth.max(), depth.min())

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
        # depthmap_bev(x, depth)

        x = self.conv2(x, depth)
        x = self.bn2(x)
        x = F.relu(x)

        # depthmap_bev(x, depth)
        # x = self.conv3(x, depth)
        # depth = F.max_pool2d(depth.float(), 3, padding=1, stride=(1,2)).long()
        # x = self.bn3(x)
        # x = F.relu(x)
        # x = self.conv4(x, depth)
        # x = self.bn4(x)
        # x = F.relu(x)
        # x = self.conv16(x, depth)
        # x = self.bn16(x)
        # x = F.relu(x)
        xs = self.conv5([x], depth)
        depth = F.avg_pool2d(depth, 1, padding=0, stride=(2,2))
        depth = depth / 2
        print("2 depth max min=", depth.max(), depth.min())
        xs = [ self.bn5(x) for x in xs]
        xs = [F.relu(x) for x in xs]
        xs = self.conv6(xs, depth)
        xs = [self.bn6(x) for x in xs]
        xs = [F.relu(x) for x in xs]
        xs = self.conv7(xs, depth)
        xs = [self.bn7(x) for x in xs]
        xs = [F.relu(x) for x in xs]


        # x = self.conv8(x, depth)
        # x = self.bn8(x)
        # x = F.relu(x)
        xs = self.conv9(xs, depth)
        depth = F.avg_pool2d(depth, 2, padding=0, stride=(2,2))
        depth = depth / 2
        print("3 depth max min=", depth.max(), depth.min())

        xs = [self.bn9(x) for x in xs]
        xs = [F.relu(x) for x in xs]
        xs = self.conv20(xs, depth)
        xs = [self.bn20(x) for x in xs]
        xs = [F.relu(x) for x in xs]
        # with torch.no_grad():
        #     depthmap_bev(x, depth)
        xs = self.conv21(xs, depth)
        xs = [self.bn21(x) for x in xs]
        xs = [F.relu(x) for x in xs]
        xs = self.conv22(xs, depth)
        xs = [self.bn22(x) for x in xs]
        xs = [F.relu(x) for x in xs]
        xs = self.conv11(xs, depth)
        depth = F.avg_pool2d(depth, 2, padding=0, stride=(2,2))
        depth = depth / 2
        print("4 depth max min=", depth.max(), depth.min())
        xs = [self.bn11(x) for x in xs]
        xs = [F.relu(x) for x in xs]
        xs = self.conv12(xs, depth)
        xs = [self.bn12(x) for x in xs]
        xs = [F.relu(x) for x in xs]
        xs = self.conv13(xs, depth)
        xs = [self.bn13(x) for x in xs]
        xs = [F.relu(x) for x in xs]
        # x = self.conv14(x, depth)
        # depth = depth // 2
        # x = self.bn14(x)
        # x = F.relu(x)
        xs = self.conv15(xs, depth)
        depth = F.avg_pool2d(depth, (2,1), padding=0, stride=(2,1))
        xs = [self.bn15(x) for x in xs]
        xs = [F.relu(x) for x in xs]
        xs = self.conv16(xs, depth)
        depth = F.avg_pool2d(depth, (2,1), padding=0, stride=(2,1))
        xs = [self.bn16(x) for x in xs]
        xs = [F.relu(x) for x in xs]
        # x = self.conv17(x, depth)
        # depth = F.max_pool2d(depth.float(), 3, padding=1, stride=(2,1)).long()
        # x = self.bn17(x)
        # x = F.relu(x)

        print("5 depth max min=", depth.max(), depth.min())
        x = depth_to_3D_v2(xs, depth, 64)
        B, C, D, H, W = x.shape
        # with torch.no_grad():
        #     # f =
        #     for c in range(C):
        #         save_2d_jpg(x[0][c].sum(dim=0), "bevmap-%d.jpg"%(c))

        print("shape BCDHW, ", B,C,D,H,W)
        return x.permute(0,1,3,4,2).reshape(B, C*H, W, D)