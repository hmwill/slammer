// BSD 3-Clause License
//
// Copyright (c) 2021, 2022, Hans-Martin Will
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice, this
//    list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
//
// 3. Neither the name of the copyright holder nor the names of its
//    contributors may be used to endorse or promote products derived from
//    this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
// FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#include <cstdlib>
#include <cmath>

#include "slammer/slammer.h"
#include "slammer/loris/opencv_utils.h"

#include "opencv2/opencv.hpp"
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/viz.hpp"

using namespace slammer;
using namespace slammer::loris;


const std::string kDataSet { "data/cafe1-1"s };

const std::string kImage1Color { "data/cafe1-1/color/1560004885.446172.png"s };
const std::string kImage1Depth { "data/cafe1-1/aligned_depth/1560004885.446165.png"s };
const std::string kImage2Color { "data/cafe1-1/color/1560004888.047959.png"s };
const std::string kImage2Depth { "data/cafe1-1/aligned_depth/1560004888.047952.png"s };


Result<cv::Mat> GetImage(const std::string& path) {
    auto image = cv::imread(path, cv::IMREAD_UNCHANGED);

    if (image.empty()) {
        std::string error_message = "Could not read image file " + path;
        return Error(error_message);
    } else {
        return image;
    }
}

Result<cv::Matx33d> GetCameraIntrinsicMatrix() {
    auto maybe_info = ReadSensorInfo(kDataSet);

    if (maybe_info.failed()) {
        return maybe_info.error();
    }

    const auto& intrinsics = maybe_info.value().d400_color_optical_frame.intrinsics;
    auto result = cv::Matx33d::zeros();
    result(0, 0) = intrinsics.at<double>(0);
    result(0, 2) = intrinsics.at<double>(1);
    result(1, 1) = intrinsics.at<double>(2);
    result(1, 2) = intrinsics.at<double>(3);
    result(2, 2) = 1.0;

    return result;
}

Result<cv::viz::WCloud> CreatePointCloudFromImages(const cv::Matx33d& camera_matrix, const cv::Mat& rgb, 
                                                   const cv::Mat& depth) {
    std::vector<cv::Point3d> coords;
    std::vector<cv::Vec3b> colors;

    auto transform = camera_matrix.inv();

    if (rgb.rows != depth.rows || rgb.cols != depth.cols) {
        return Error("Incompatible image size for point cloud creation");
    }

    for (int row = 0; row < rgb.rows; ++row) {
        for (int column = 0; column < rgb.cols; ++column) {
            auto z = static_cast<double>(depth.at<ushort>(row, column));
            cv::Point3d image_coord(column * z, row * z, z);
            auto world_coord = transform * image_coord;
            coords.push_back(world_coord);

            auto color = rgb.at<cv::Vec3b>(row, column);
            colors.push_back(color);
        }
    }

	return cv::viz::WCloud(coords, colors);
}

int main(int argc, char *argv) {
    auto maybe_image1_color = GetImage(kImage1Color);
    auto maybe_image1_depth = GetImage(kImage1Depth);

    auto maybe_image2_color = GetImage(kImage2Color);
    auto maybe_image2_depth = GetImage(kImage2Depth);

    if (maybe_image1_color.failed() || maybe_image1_depth.failed() ||
        maybe_image2_color.failed() || maybe_image2_depth.failed()) {
        std::cerr << "Could not load image files" << std::endl;
        return EXIT_FAILURE;
    }

    const auto& image1_color = maybe_image1_color.value();
    const auto& image1_depth = maybe_image1_depth.value();
    const auto& image2_color = maybe_image2_color.value();
    const auto& image2_depth = maybe_image2_depth.value();

    auto maybe_camera_matrix = GetCameraIntrinsicMatrix();

    if (maybe_camera_matrix.failed()) {
        std::cerr << "Could not load camera matrix" << std::endl;
        return EXIT_FAILURE;
    }

    cv::Matx33d camera_matrix = maybe_camera_matrix.value();

    // Create a point cloud from a color/depth pair
    std::vector<cv::Point3d> coords;
    std::vector<cv::viz::Color> colors;

    auto maybe_cloud = CreatePointCloudFromImages(camera_matrix, image1_color, image1_depth);

    if (maybe_cloud.failed()) {
        std::cerr << "Could not create point cloud" << std::endl;
        return EXIT_FAILURE;
    }

    auto cloud = maybe_cloud.value();
	cloud.setRenderingProperty(cv::viz::POINT_SIZE, 3.);

    cv::viz::Viz3d visualizer("Viewer");
    visualizer.setBackgroundColor(cv::viz::Color::black());

    visualizer.showWidget("Cloud", cloud);
	
    while (!visualizer.wasStopped()) {
        visualizer.spinOnce(/*1, true*/);
    }

    return EXIT_SUCCESS;
}