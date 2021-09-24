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

#include <gtest/gtest.h>

#include "slammer/loris/opencv_utils.h"

using namespace slammer;
using namespace slammer::loris;


TEST(SlammerLorisTest, OpenCvUtils_ReadSensorInfo) {
    auto result = ReadSensorInfo("data/cafe1-1");

    EXPECT_TRUE(result.ok());
    const auto& sensor_info = result.value();

    EXPECT_EQ(sensor_info.d400_color_optical_frame.fps, 30);
    EXPECT_EQ(sensor_info.d400_color_optical_frame.sensor_name, "d400_color_optical_frame");
    EXPECT_EQ(sensor_info.d400_color_optical_frame.width, 848);
    EXPECT_EQ(sensor_info.d400_color_optical_frame.height, 480);
    EXPECT_EQ(sensor_info.d400_color_optical_frame.distortion_model, "radial-tangential");
    EXPECT_EQ(sensor_info.d400_color_optical_frame.intrinsics.rows, 1);
    EXPECT_EQ(sensor_info.d400_color_optical_frame.intrinsics.cols, 4);
    EXPECT_EQ(sensor_info.d400_color_optical_frame.intrinsics.at<double>(0, 1), 4.3320397949218750e+02);
    EXPECT_EQ(sensor_info.d400_color_optical_frame.model, "pinhole");
    EXPECT_EQ(sensor_info.d400_color_optical_frame.distortion_coefficients.rows, 1);
    EXPECT_EQ(sensor_info.d400_color_optical_frame.distortion_coefficients.cols, 5);
    EXPECT_EQ(sensor_info.d400_color_optical_frame.distortion_coefficients.at<double>(0, 2), 0.0);

    EXPECT_EQ(sensor_info.t265_gyroscope.fps, 200);
    EXPECT_EQ(sensor_info.t265_gyroscope.sensor_name, "t265_gyroscope");
    EXPECT_EQ(sensor_info.t265_gyroscope.imu_intrinsic.rows, 1);
    EXPECT_EQ(sensor_info.t265_gyroscope.imu_intrinsic.cols, 12);
    EXPECT_EQ(sensor_info.t265_gyroscope.imu_intrinsic.at<double>(0, 0), 9.9567949771881104e-01);
    EXPECT_EQ(sensor_info.t265_gyroscope.noise_variances.rows, 1);
    EXPECT_EQ(sensor_info.t265_gyroscope.noise_variances.cols, 3);
    EXPECT_EQ(sensor_info.t265_gyroscope.noise_variances.at<double>(0, 2), 5.1480301408446394e-06);
    EXPECT_EQ(sensor_info.t265_gyroscope.bias_variances.rows, 1);
    EXPECT_EQ(sensor_info.t265_gyroscope.bias_variances.cols, 3);
    EXPECT_EQ(sensor_info.t265_gyroscope.bias_variances.at<double>(0, 1), 4.9999999873762135e-07);

    EXPECT_EQ(sensor_info.odometer.fps, 0);
    EXPECT_EQ(sensor_info.odometer.sensor_name, "odometer");
}

TEST(SlammerLorisTest, OpenCvUtils_ReadFrames) {
    auto result = ReadFrames("data/cafe1-1");

    EXPECT_TRUE(result.ok());
    const auto& transformations = result.value();
    EXPECT_EQ(transformations.size(), 9);

    auto frame_iter = transformations.find("d400_depth_optical_frame");
    EXPECT_NE(frame_iter, transformations.end());
    EXPECT_EQ(frame_iter->second.name, "d400_depth_optical_frame");
    EXPECT_EQ(frame_iter->second.parent_name, "d400_color_optical_frame");
    EXPECT_EQ(frame_iter->second.transformation.rows, 4);
    EXPECT_EQ(frame_iter->second.transformation.cols, 4);
    EXPECT_EQ(frame_iter->second.transformation.at<double>(1, 2), -3.5019440527251228e-03);
}

