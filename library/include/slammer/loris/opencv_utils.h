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

#ifndef SLAMMER_LORIS_OPENCV_UTILS_H
#define SLAMMER_LORIS_OPENCV_UTILS_H

#pragma once

#include "slammer/slammer.h"
#include "slammer/camera.h"

#include <string>
#include <unordered_map>

#include "opencv2/opencv.hpp"


namespace slammer {
namespace loris {

struct CommonParameters {
    std::string sensor_name;
    int fps;
};

struct CameraParameters: public CommonParameters {
    int width;
    int height;
    std::string model;
    cv::Mat intrinsics;
    std::string distortion_model;
    cv::Mat distortion_coefficients;
};

struct ImuParameters: public CommonParameters {
    cv::Mat imu_intrinsic;
    cv::Mat noise_variances;
    cv::Mat bias_variances;
};

struct OdometerParameters: public CommonParameters {};

/// The sensor calibration information provided with the data sets
struct SensorInfo {
    CameraParameters d400_color_optical_frame;
    CameraParameters d400_depth_optical_frame;
    CameraParameters t265_fisheye1_optical_frame;
    CameraParameters t265_fisheye2_optical_frame;
    ImuParameters d400_accelerometer;
    ImuParameters d400_gyroscope;
    ImuParameters t265_accelerometer;
    ImuParameters t265_gyroscope;
    OdometerParameters odometer;
};

/// Read the sensor information for the dataset identified by the provided path
Result<SensorInfo> ReadSensorInfo(const std::string& dataset_path);

using FrameName = std::string;

struct Frame {
    FrameName name;
    FrameName parent_name;
    cv::Mat transformation;
};

extern const std::string kBaseLink;

using FrameSet = std::unordered_map<FrameName, Frame>;

/// Read the frame transformation information for the dataset identified by the provided path
Result<FrameSet> ReadFrames(const std::string& transformations_path);

/// Create a camera based on camera parameters and a transformation describing the pose
///
/// \param parameters       camera parameters as maintained by OpenCV
/// \param pose             the camera pose relative to the robot frame
///
/// \return                 a camera object for the SLAM enginer
Camera CreateCamera(const CameraParameters& parameters, const SE3d& pose);

/// Retrieve the pose relative to origin (base_link) for a specific frame
///
/// \param frames   the frame set describing relative poses of frames to each other
/// \param name     the name of the specific frame whose absolute pose we want to retrieve
/// \return         the pose relative to the origin (base_link) or a failure if the name could not be fully resolved
Result<SE3d> GetFramePose(const FrameSet& frames, const FrameName& name);

} // namespace loris
} // namespace slammer

#endif //ndef SLAMMER_LORIS_OPENCV_UTILS_H
