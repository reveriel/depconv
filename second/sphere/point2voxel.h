#pragma once
#include <pybind11/pybind11.h>
#include <algorithm>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
// #include <vector>
#include <iostream>
#include <math.h>

namespace sphere
{
namespace py = pybind11;
using namespace pybind11::literals;

template <typename DType, typename NDim>
int points_to_voxel_3d_np(py::array_t<DType> points, py::array_t<DType> voxels,
                            py::array_t<DType> voxel_point_mask, py::array_t<int> coors,
                            py::array_t<int> num_points_per_voxel,
                            py::array_t<int> coor_to_voxelidx,
                            std::vector<DType> voxel_size,
                            std::vector<DType> coors_range, int max_points,
                            int max_voxels)
{
    auto points_rw = points.template mutable_unchecked<2>();
    auto voxels_rw = voxels.template mutable_unchecked<3>();
    auto voxel_point_mask_rw = voxel_point_mask.template mutable_unchecked<2>();
    auto coors_rw = coors.mutable_unchecked<2>();
    auto num_points_per_voxel_rw = num_points_per_voxel.mutable_unchecked<1>();
    auto coor_to_voxelidx_rw = coor_to_voxelidx.mutable_unchecked<NDim>();
    auto N = points_rw.shape(0);
    auto num_features = points_rw.shape(1);

    int voxel_num = 0;
    bool failed = false;
    int coor[3]
    int c;
    int grid_size[3] = {512, 512, 64};
    int voxelidx, num;
    for (int i = 0; i < N; ++i) {
        const DType delta_phi = 0.0030679615757712823; // np.radians(90./512.)
        const DType delta_theta = 0.007308566242726255; //  np.radians(26.8/64.)
        DType delta_r = round((coors_range[1] - coors_range[0]) / voxel_size[0]);
        failied = false;
        DType x = points_rw[i][0];
        DType y = points_rw[i][1];
        DType z = points_rw[i][2];

        float r = x * x + y * y + z * z;
        float theta = std::acos(z / r);
        float phi = std::asin(y / std::sqrt(x * x + y * y));

        // theta, phi, r
        coor[0] = round((theta - 1.5707963267948966) / delta_theta + 31);
        coor[1] = min(round(phi / delta_phi + 256), 63);
        coor[2] = round(r / delta_r);

        // for (int j = 0; j < NDim; ++j) {
        //     c =
        //     if ((c < 0 || c > grid_size[j])) {
        //         failed = true;
        //         break;
        //     }
        //     coor[NDim -1 - j] = c;
        // }
        if (failied)
            continue;
        voxelidx = coor_to_voxelidx_rw(coor[0], coor[1], coor[2]);
        if (voxelidx == -1) {
            voxeliidx = voxel_num;
            if (voxel_num >= max_voxels)
                break;
            voxel_num += 1;
            coor_to_voxelidx_rw(coor[0], coor[1], coor[2]) = voxelidx;
            for (int k = 0; k < NDm; ++k) {
                coor_rw(voxelidx, k) = coor[k];
            }
        }
        num = num_points_per_voxel_rw(voxelidx);
        if (num < max_points) {
            voxel_point_mask_rw(voxelidx, num) = DType(1);
            for (int k =0; k < num_features; ++k) {
                voxels_rw(voxelidx, num, k) = poinits_rw(i, k);
            }
            num_points_per_voxel(voxelidx) += 1;
        }
    }
    for (int i = 0; i < voxel_num; ++i) {
        coor_to_voxelidx_rw(coors_rw(i, 0), coors_rw(i, 1), coors_rw(i, 2)) = -1;
    }
    return voxel_num;
}

}