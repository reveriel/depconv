
import numpy as np
import math

import torch
import torch.nn.functional as F
from torch import nn
from typing import Dict, List, Tuple, Union


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
    unfolded_depth = F.unfold(depth_, kernel_size=(kH, kW),
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

def init_depth_from_feature(feature, k, range=[0, 70]):
    """
        return a depth tensor of shape (B, H, W)
        max depth be 'k-1'

    Argument:
        feature: tensor of shape(B, C, H, W),
            C = 5
    """
    r = feature[:, 3, :, :]
    depth = ((r - range[0]) * (k-1) / (range[1]- range[0]))
    # some are out of range
    depth[depth >= k] = k-1
    return depth


class DepConv3D_v2(DepConv3D):
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
        super(DepConv3D_v2, self).__init__(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            subm=subm)

    def forward(self, features, depth):
        out_tensor = depconv3d_v2(features, depth, self.weight,
            self.bias, self.stride, self.padding, self.dilation,
            self.groups)
        if self.subm:
            out_tensor = out_tensor[1:-1]
        return out_tensor

def depconv3d_v2(input:List[torch.Tensor],
        depth:torch.Tensor,
        weight:torch.Tensor,
        bias=torch.tensor([]),
        stride=1,
        padding=0,
        dilation=1,
        groups=1
        ):
    """
    Applies a 3D convolution on 2D input with depth info.

    Parameters:

        input: the input is a list of  tensors of shape (minibatch, in_channels, iH, iW)

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

        a list of tensors of shape(N, out_channels, oH, oW)

        where

            oH = floor((iH + 2 x padH - dH x (kH - 1) -1) / sH + 1)
            oW = floor((iW + 2 x padW - dW x (kW - 1) -1) / sW + 1)

    Examples:

        filters = torch.torch.randn(33, 16, 3, 3)
        depth = torch.randn()
    """
    # group not supported yet
    assert groups == 1
    assert input[0].dim() == 4
    assert depth.dim() == 3
    assert weight.dim() == 5

    if not isinstance(stride, (list, tuple)):
        stride = [stride] * 3
    if not isinstance(padding, (list, tuple)):
        padding = [padding] * 3
    if not isinstance(dilation, (list, tuple)):
        dilation = [dilation] * 3

    return depconv3d_v2_(input, depth, weight,
        bias, stride, padding, dilation, groups)



# @torch.jit.script
def depconv3d_v2_(input:List[torch.Tensor],
        depth, weight,
        bias:torch.Tensor,
        stride:List[int],
        padding:List[int],
        dilation:List[int],
        groups:int):

    B, iC, iH, iW = input[0].shape
    oC, iC, kD, kH, kW = weight.shape
    sD, sH, sW = stride
    padD, padH, padW = padding
    dD, dH, dW = dilation
    assert (kD == 3), "only support weight of shape 3xnxn now"
    assert (dD == 1 and dH == 1 and dW == 1), "only support dilation of 1x1x1"

    # im2colo for tensors in inputs
    def unfold_inputs(inputs:List[torch.Tensor]):
        res = []
        for i in inputs:
            # im2colo
            # unfold, only 4D input tensors are supported
            unfolded = F.unfold(i, kernel_size=(kH, kW), dilation=dilation[1:], padding=padding[1:], stride=stride[1:])
            # B, iC * kH * kW, N
            res.append(unfolded)
        return res

    def get_indexes(depth:torch.Tensor):
        """
        return the index and 'N', the number of sliding window
        """
        depth_ = depth.unsqueeze(1) # B, 1, iH, iW
        # unfold only support float tensor
        unfolded_depth = F.unfold(depth_, kernel_size=(kH, kW),
            dilation=dilation[1:], padding=padding[1:], stride=stride[1:]).long()
         # B , kH * kW * 1 , N
        N = unfolded_depth.size(-1)

        # depth difference
        depth_diff = (unfolded_depth - unfolded_depth[:, kH * kW // 2, :].unsqueeze(1))
        # B, kH * kW * 1, N

        # indexes
        i0, i1, i2 = depth_diff, depth_diff + 1, depth_diff + 2

        return clip_depth_index(i0), clip_depth_index(i1), clip_depth_index(i2), N
        # B, kH*kW*1 , N

    # in range
    def clip_depth_index(index):
        index[index < 0] = 3
        index[index > 3] = 3
        return index


    def unfold_weight(index, weght, N):
        # extend weight with zeros, for indexing out of range values
        ext_weight = torch.cat((weight, torch.zeros([oC, iC, 1, kH, kW], device=torch.device('cuda'))), dim=2)
            # oC, iC, kD+1, kH, kW

        index = index.reshape(B, 1, 1, 1, kH *kW, N).expand(-1, oC, iC, -1, -1, -1) # B, oC, iC, 1, kH*kW, N
        b_weight = ext_weight.reshape(oC, iC, kD + 1, kH * kW).unsqueeze(0) \
            .unsqueeze(-1).expand(B, -1, -1, -1, -1, N)  # B, oC, iC, kD, kH*kW, N

        unfolded_weight = b_weight.gather(3, index).reshape(B, oC, iC * kH * kW, N) # B, oC, iC * kH * kW, N
        return unfolded_weight

    i0, i1, i2, N = get_indexes(depth)

    w0 = unfold_weight(i0, weight, N)
    w1 = unfold_weight(i1, weight, N)
    w2 = unfold_weight(i2, weight, N)

    def mul_(weight, feature):
        return torch.einsum('bikj,bkj->bij', weight, feature)
        # B, oC, N

    unfolded_inputs = unfold_inputs(input)

    out0 = [ mul_(w2, f) for f in unfolded_inputs] + [0, 0]
    out1 = [0] + [ mul_(w1, f) for f in unfolded_inputs] + [0]
    out2 = [0,0] + [ mul_(w0, f) for f in unfolded_inputs]

    out = [a + b + c for a, b, c in zip(out0, out1, out2)]

    # print("type",(unfolded_weight.dtype), (unfolded_input.dtype))
    # print("enisum shape ", unfolded_weight.shape, unfolded_input.shape)
    # unfolded_output = torch.einsum('bikj,bkj->bij', unfolded_weight, unfolded_input)
    # # B, oC, N

    oH = int(math.floor((iH + 2 * padH - dH * (kH - 1) -1) / sH + 1))
    oW = int(math.floor((iW + 2 * padW - dW * (kW - 1) -1) / sW + 1))
    #

    output = [ F.fold(o, output_size=(oH, oW), kernel_size=(1,1)) for o in out]
    # change the depth
    # depth = ((depth + padD - dD * (kD // 2)) // sD + 1).long()
    # depth

    # if bias is not None:
    #     output += bias.unsqueeze(0).expand_as(output)
    # output = output.reshape()
    return output

def test_depconv3dv2():
    feature = [torch.tensor(
        [[.1,.2,.3],
        [.4,.5,.6],
        [.7,.8,.9]]).reshape(1,1,3,3).cuda()]
    depth = torch.tensor(
        [[0,0,1],
         [1,1,0],
         [1,1,2]]).reshape(1,3,3).cuda()
    depth = torch.tensor(
        [[1,1,1],
         [1,1,1],
         [1,1,1]]).reshape(1,3,3).cuda()
    depth = torch.tensor(
        [[0,0,0],
         [1,1,1],
         [2,2,2]]).reshape(1,3,3).cuda()
    weight = torch.ones(1,1,3,3,3).cuda()
    res = depconv3d_v2(feature, depth, weight, padding=1)


    feature_dense = depth_to_3D(feature[0], depth, 3)
    print(feature_dense)
    res_ref = F.conv3d(feature_dense, weight, padding=1)


    # res = (depth_to_3D(res[0],depth, 5) +
    #     depth_to_3D(res[1], depth+1, 5) +
    #     depth_to_3D(res[2], depth+2, 5))
    res = depth_to_3D_v2(res, depth, 3)
    print(res)

    print(res_ref)

# test_depconv3dv2()


def depth_to_2D(feature, depth, D=0):
    """
    extract one thin layer, of the 3D voxels

    Aruguments:
        feature : tensor of shape (batchsize, Channel, H, W)
        depth : long tensor of shape (batchsize, H, W)
        D: index of the extracted slice

    Returns:
        a 3D tensor of size (batchsize, Channel, D, H, W)
    """
    assert False, "unfinished, untested"
    B, C, H, W = feature.shape
    device = feature.get_device()
    ret_tensor = torch.zeros(B, C, D, H, W, device=device)
    feature_ = feature.reshape(B, C, 1, H, W).expand_as(ret_tensor)
    depth_idx = depth.reshape(B, 1, 1, H, W).expand_as(ret_tensor)
    depth_idx[depth_idx >= D]  = D - 1
    # expand to shape, B, C, D, H, W
    ret_tensor.scatter_(2, depth_idx, feature_)
    return ret_tensor

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

    D_min, D_max = depth.min(), depth.max()
    print("D_min, D_max= ", D_min, D_max)
    if D == 0:
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


def depth_to_3D_v2(feature:list, depth, D=0):
    """
    convert (depth, feature) to a 3D tensor

    Aruguments:
        feature : a list of tensor of shape (batchsize, Channel, H, W)
        depth : long tensor of shape (batchsize, H, W)

    Returns:
        a 3D tensor of size (batchsize, Channel, D, H, W)
    """
    # if not isinstance(depth, torch.LongTensor):
    #      raise ValueError("'depth' should be of type LongTensor,"
    #         " but got %s" % type(depth))
    thick = len(feature)
    mid = thick//2
    depth = depth.long()

    D_min, D_max = depth.min(), depth.max()
    print("D_min, D_max= ", D_min, D_max)
    if D == 0:
        D = D_max - D_min + 1


    B, C, H, W = feature[0].shape
    # print("feature", feature)
    device = feature[0].get_device()
    D = D + mid + mid
    ret_tensor = torch.zeros(B, C, D, H, W, device=device)
    # print("BCDHW,", B,C,D,H,W)
    depth_idx = depth.reshape(B, 1, 1, H, W).expand_as(ret_tensor)

    for i in range(thick):
        f_i = feature[i].reshape(B, C, 1, H, W).expand_as(ret_tensor)
        idx_i = depth_idx + i
        idx_i[idx_i > D] = D-1
        ## ??? why have to do this ?
        ret_tensor.scatter_(2, idx_i, f_i)
        # print("ret", ret_tensor)
    return ret_tensor[:,:,mid:-mid,:,:]



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





# _test_depth_to_3D()
# test_submanifold_conv3d()
# test_xyz2range_v2()
