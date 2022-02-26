cmake_minimum_required(VERSION 3.21)
project(slammer)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
# The following define enables c-style math macros in C++; used by Sophus
add_definitions(-D_USE_MATH_DEFINES)

enable_testing()
find_package(GTest REQUIRED)
include(Dart)

find_package(absl CONFIG REQUIRED)
include_directories(${absl_INCLUDE_DIRS})

find_package(Boost REQUIRED)
include_directories(${Boost_INCLUDE_DIRS})

find_package(Eigen3 3.3 REQUIRED)
include_directories(${Eigen3_INCLUDE_DIRS})

find_package(Sophus REQUIRED)
include_directories(${Sophus_INCLUDE_DIRS})

find_package(Pangolin CONFIG REQUIRED)
include_directories(${Pangolin_INCLUDE_DIRS})

find_package(OpenCV 4.5 REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

find_package(SuiteSparse CONFIG REQUIRED)
include_directories(${SuiteSparse_INCLUDE_DIRS})

find_package(g2o REQUIRED)
include_directories(${g2o_INCLUDE_DIRS})

find_package(fmt REQUIRED)
include_directories(${fmt_INCLUDE_DIRS})

find_package(libpng REQUIRED)
include_directories(${libpng_INCLUDE_DIRS})

# Incorporating rapidcsv directly because vcpkg/CMake interation non-functioning
#find_package(rapidcsv REQUIRED)
include_directories(third-party/rapidcsv)

add_subdirectory(library)
add_subdirectory(tools)