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

#include <stdexcept>

#include <gtest/gtest.h>

#include "slammer/pipeline.h"
#include "slammer/loris/driver.h"
#include "slammer/loris/opencv_utils.h"

using namespace slammer;
using namespace slammer::loris;


TEST(PipelineTest, RunPipeline) {
    using namespace std::placeholders;

    std::string kDataSetPath("data/cafe1-1");

    Result<SensorInfo> sensor_info_result = ReadSensorInfo(kDataSetPath);
    EXPECT_TRUE(sensor_info_result.ok());
    auto sensor_info = sensor_info_result.value();

    Result<FrameSet> frame_info_result = ReadFrames(kDataSetPath);
    EXPECT_TRUE(frame_info_result.ok());
    auto frame_info = frame_info_result.value();

    Driver driver(kDataSetPath);

    auto rgb_camera_pose = GetFramePose(frame_info, "d400_color_optical_frame");
    EXPECT_TRUE(rgb_camera_pose.ok());
    auto depth_camera_pose = GetFramePose(frame_info, "d400_depth_optical_frame");
    EXPECT_TRUE(depth_camera_pose.ok());

    Camera rgb_camera = CreateCamera(sensor_info.d400_color_optical_frame, rgb_camera_pose.value());
    //Camera depth_camera = CreateCamera(sensor_info.d400_depth_optical_frame, depth_camera_pose.value());
    StereoDepthCamera depth_camera = CreateAlignedStereoDepthCamera(sensor_info.d400_depth_optical_frame, 0.05, SE3d());

    Vocabulary vocabulary;

    FrontendParameters frontend_parameters;
    Backend::Parameters backend_parameters;

    Pipeline pipeline(frontend_parameters, backend_parameters,
        std::move(vocabulary), std::move(rgb_camera), std::move(depth_camera),
        driver.color, driver.aligned_depth);

    // run for 2 secs of simulated events
    auto result = driver.Run(slammer::Timediff(2.0));
    EXPECT_TRUE(result.ok());
}