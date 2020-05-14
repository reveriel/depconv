cmake_minimum_required(VERSION 3.7 FATAL_ERROR)


project(cmake_example)

add_subdirectory(pybind11)
pybind11_add_module(cmake_example transform.cc)