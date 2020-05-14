#include "point2voxel.h"

namespace py = pybind11;
using namespace pybind11::literals;

PYBIND11_MODULE(sphere_transform, m) {
    m.doc() = "sphere point to voxel";
    m.def("points_to_voxel_3d_np", &sphere::points_to_voxel_3d_np<float, 3>,
    "matrix tensor_square", "points"_a = 1, "voxels"_a = 2,
    "voxel_point_mask"_a = 3, "coors"_a = 4, "num_points_per_voxel"_a = 5,
    "coor_to_voxelidx"_a = 6, "voxel_size"_a = 7, "coors_range"_a = 8,
    "max_points"_a = 9, "max_voxels"_a = 10);
}