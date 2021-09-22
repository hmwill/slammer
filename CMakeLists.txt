cmake_minimum_required(VERSION 3.21)
project(slammer)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

enable_testing()
find_package(GTest REQUIRED)
include(Dart)

find_package(Arrow 5.0 REQUIRED)
include_directories(${Arrow_INCLUDE_DIRS})

find_package(Eigen3 3.3 REQUIRED)
include_directories(${Eigen3_INCLUDE_DIRS})

find_package(Sophus REQUIRED)
include_directories(${Sophus_INCLUDE_DIRS})

find_package(Pangolin REQUIRED)
include_directories(${Pangolin_INCLUDE_DIRS})

find_package(OpenCV 4.5 REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_subdirectory(library)
add_subdirectory(tools)