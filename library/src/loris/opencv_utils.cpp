// BSD 3-Clause License
//
// Copyright (c) 2021, Hans-Martin Will
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

#include "slammer/loris/opencv_utils.h"

#include <optional>

using namespace slammer;
using namespace loris;

namespace {

std::optional<Error> ReadCommonParameters(const cv::FileNode& node, CommonParameters& common) {
    node["sensor_name"] >> common.sensor_name;
    node["fps"] >> common.fps;
    return {};
}

std::optional<Error> ReadCameraParameters(const cv::FileNode& node, CameraParameters& camera) {
    if (!node.isMap()) {
        return Error("Expected a map to describe parameters");
    }

    if (static_cast<std::string>(node["sensor_type"]) != "camera") {
        return Error("Invalid node type for camera parameters");
    }

    if (auto maybe_error = ReadCommonParameters(node, camera)) {
        return maybe_error;
    }

    node["width"] >> camera.width;
    node["height"] >> camera.height;
    node["model"] >> camera.model;
    node["intrinsics"] >> camera.intrinsics;
    node["distortion_model"] >> camera.distortion_model;
    node["distortion_coefficients"] >> camera.distortion_coefficients;

    return {};
}

std::optional<Error> ReadImuParameters(const cv::FileNode& node, ImuParameters& imu) {
    if (!node.isMap()) {
        return Error("Expected a map to describe parameters");
    }

    if (static_cast<std::string>(node["sensor_type"]) != "imu") {
        return Error("Invalid node type for camera parameters");
    }

    if (auto maybe_error = ReadCommonParameters(node, imu)) {
        return maybe_error;
    }

    node["imu_intrinsic"] >> imu.imu_intrinsic;
    node["noise_variances"] >> imu.noise_variances;
    node["bias_variances"] >> imu.bias_variances;

    return {};
}

std::optional<Error> ReadOdometerParameters(const cv::FileNode& node, OdometerParameters& odometer) {
    if (!node.isMap()) {
        return Error("Expected a map to describe parameters");
    }

    if (static_cast<std::string>(node["sensor_type"]) != "odom") {
        return Error("Invalid node type for camera parameters");
    }

    return ReadCommonParameters(node, odometer);
}

std::optional<Error> ReadFrame(const cv::FileNode& node, Frame& frame) {
    if (!node.isMap()) {
        return Error("Expected a map to describe parameters of a frame transformation");
    }

    node["parent_frame"] >> frame.parent_name;
    node["child_frame"] >> frame.name;
    node["matrix"] >> frame.transformation;

    return {};
}

} // namespace 

Result<SensorInfo> slammer::loris::ReadSensorInfo(const std::string& dataset_path) {
    SensorInfo result;

    cv::FileStorage fs;
    std::string full_path = dataset_path + "/sensors.yaml";
    if (!fs.open(full_path, cv::FileStorage::READ)) {
        std::string message = "Could not open YAML file " + full_path; 
        return Error(message);
    }

    if (auto maybe_error = ReadCameraParameters(fs["d400_color_optical_frame"], result.d400_color_optical_frame)) {
        return maybe_error.value();
    }

    if (auto maybe_error = ReadCameraParameters(fs["d400_depth_optical_frame"], result.d400_depth_optical_frame)) {
        return maybe_error.value();
    }

    if (auto maybe_error = ReadCameraParameters(fs["t265_fisheye1_optical_frame"], result.t265_fisheye1_optical_frame)) {
        return maybe_error.value();
    }

    if (auto maybe_error = ReadCameraParameters(fs["t265_fisheye2_optical_frame"], result.t265_fisheye2_optical_frame)) {
        return maybe_error.value();
    }

    if (auto maybe_error = ReadImuParameters(fs["d400_accelerometer"], result.d400_accelerometer)) {
        return maybe_error.value();
    }

    if (auto maybe_error = ReadImuParameters(fs["d400_gyroscope"], result.d400_gyroscope)) {
        return maybe_error.value();
    }

    if (auto maybe_error = ReadImuParameters(fs["t265_accelerometer"], result.t265_accelerometer)) {
        return maybe_error.value();
    }

    if (auto maybe_error = ReadImuParameters(fs["t265_gyroscope"], result.t265_gyroscope)) {
        return maybe_error.value();
    }

    if (auto maybe_error = ReadOdometerParameters(fs["odometer"], result.odometer)) {
        return maybe_error.value();
    }

    return result;
}



Result<FrameSet> slammer::loris::ReadFrames(const std::string& transformations_path) {
    FrameSet result;

    cv::FileStorage fs;
    std::string full_path = transformations_path + "/trans_matrix.yaml";
    if (!fs.open(full_path, cv::FileStorage::READ)) {
        std::string message = "Could not open YAML file " + full_path; 
        return Error(message);
    }

    auto frame_list = fs["trans_matrix"];
    if (!frame_list.isSeq()) {
        std::string message = "Missing list of transformation frames in YAML file " + full_path; 
        return Error(message);
    }

    for (auto iter = frame_list.begin(), end = frame_list.end(); iter != end; ++iter) {
        Frame frame;

        if (auto maybe_error = ReadFrame(*iter, frame)) {
            return maybe_error.value();
        }

        result[frame.name] = frame;
    }

    // validation; all parent frames other than "base_link" should have been provided

    for (const auto& element: result) {
        if (element.second.parent_name != kBaseLink) {
            if (result.find(element.second.parent_name) == result.end()) {
                std::string message = "Missing parent frame for transformation " + element.second.name; 
                return Error(message);
            }
        }
    }

    return result;
}

const FrameName slammer::loris::kBaseLink { "base_link" };