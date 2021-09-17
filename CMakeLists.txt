cmake_minimum_required(VERSION 3.21)
project(slammer)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

enable_testing()
find_package(GTest REQUIRED)

find_package(Arrow 5.0 REQUIRED)
find_package(Eigen3 3.3 REQUIRED)
find_package(Sophus REQUIRED)
find_package(Pangolin REQUIRED)
find_package(OpenCV 4.5 REQUIRED)

add_subdirectory(library)
add_subdirectory(tools)