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


TEST(FrontendTest, RunFrontend) {
    using namespace std::placeholders;

    std::string kDataSetPath("data/cafe1-1");

    Result<SensorInfo> sensor_info_result = ReadSensorInfo(kDataSetPath);
    EXPECT_TRUE(sensor_info_result.ok());
    auto sensor_info = sensor_info_result.value();

    Result<FrameSet> frame_info_result = ReadFrames(kDataSetPath);
    EXPECT_TRUE(frame_info_result.ok());
    auto frame_info = frame_info_result.value();

    Driver driver(kDataSetPath, arrow::io::default_io_context());

    auto rgb_camera_pose = GetFramePose(frame_info, "d400_color_optical_frame");
    EXPECT_TRUE(rgb_camera_pose.ok());
    auto depth_camera_pose = GetFramePose(frame_info, "d400_depth_optical_frame");
    EXPECT_TRUE(depth_camera_pose.ok());

    Camera rgb_camera = CreateCamera(sensor_info.d400_color_optical_frame, rgb_camera_pose.value());
    Camera depth_camera = CreateCamera(sensor_info.d400_depth_optical_frame, depth_camera_pose.value());
    RgbdFrontend::Parameters frontend_parameters;

    // process only 1 frane/sec
    frontend_parameters.skip_count = 5;
    frontend_parameters.max_keyframe_interval = Timediff(1.0);

    RgbdFrontend frontend(frontend_parameters, rgb_camera, depth_camera);

    driver.color.AddHandler(std::bind(&RgbdFrontend::HandleColorEvent, &frontend, _1));
    driver.aligned_depth.AddHandler(std::bind(&RgbdFrontend::HandleDepthEvent, &frontend, _1));

    struct FrontendListener {
        RgbdFrontend& frontend;
        size_t num_frames, num_keyframes;
        Timestamp min_time, max_time, last_timestamp;

        Point3d groundtruth_position;
        Quaterniond groundtruth_orientation;

        // as estimated by frontend
        bool has_last_pose;
        SE3d last_keyframe_pose;

        // as provided as part of groundtruth
        Point3d last_keyframe_position;
        Quaterniond last_keyframe_orientation;

        FrontendListener(RgbdFrontend& frontend)
            : frontend(frontend), num_frames(0), num_keyframes(0), has_last_pose(false),
                min_time(Timestamp(Timediff(std::numeric_limits<double>::max()))),
                max_time(Timestamp(Timediff(std::numeric_limits<double>::min()))) {}

        void HandleFrameEvent(const ProcessedFrameEvent& event) {
            if (num_frames) {
                EXPECT_GT(event.timestamp, last_timestamp);
                last_timestamp = event.timestamp;
            }

            ++num_frames;
            num_keyframes += event.is_keyframe;

            min_time = std::min(min_time, event.timestamp);
            max_time = std::max(max_time, event.timestamp);

            auto pose = event.pose;
            auto rotation = pose.unit_quaternion();
            auto translation = pose.translation();

            EXPECT_FALSE(std::isnan(rotation.x()));
            EXPECT_FALSE(std::isnan(rotation.y()));
            EXPECT_FALSE(std::isnan(rotation.z()));
            EXPECT_FALSE(std::isnan(rotation.w()));

            EXPECT_FALSE(std::isnan(translation.x()));
            EXPECT_FALSE(std::isnan(translation.y()));
            EXPECT_FALSE(std::isnan(translation.z()));

            // Make sure we do not loose track right after establishing a new keyframe
            EXPECT_TRUE(event.new_state == RgbdFrontend::Status::kTracking ||
                        event.old_state != RgbdFrontend::Status::kNewKeyframe);

            if (event.old_state == RgbdFrontend::Status::kTracking &&
                event.new_state == RgbdFrontend::Status::kTracking) {
                EXPECT_GE(event.num_tracked_features, frontend.parameters().num_features_tracking_bad);
            }

            if (event.is_keyframe) {
                EXPECT_GE(event.num_tracked_features, frontend.parameters().num_features_tracking);

                if (has_last_pose) {
                    SE3d groundtruth_pose(groundtruth_orientation, groundtruth_position);
                    SE3d last_goundtruth_pose(last_keyframe_orientation, last_keyframe_position);
                    auto groundtruth_translation = (groundtruth_pose * last_goundtruth_pose.inverse()).translation();
                    auto groundtruth_distance = groundtruth_translation.norm();
                    auto estimated_translation = (event.pose * last_keyframe_pose.inverse()).translation();
                    auto estimated_distance = estimated_translation.norm();

                    // should be within 90%?
                    auto diff = estimated_distance - groundtruth_distance;
                    EXPECT_LE(fabs(diff), 0.1);

                    auto factor = estimated_distance / groundtruth_distance;

                    EXPECT_GE(factor, 0.9);
                    EXPECT_LE(factor, 1.1);

                    EXPECT_EQ(estimated_distance, groundtruth_distance);
                }

                last_keyframe_pose = event.pose;
                last_keyframe_position = groundtruth_position;
                last_keyframe_orientation = groundtruth_orientation;
                has_last_pose = true;
            }
        }

        void HandleGroundtruthEvent(const loris::GroundtruthEvent& event) {
            groundtruth_position = event.position;
            groundtruth_orientation = event.orientation;
        }
    };

    FrontendListener listener(frontend);
    frontend.processed_frames.AddHandler(std::bind(&FrontendListener::HandleFrameEvent, &listener, _1));
    driver.groundtruth.AddHandler(std::bind(&FrontendListener::HandleGroundtruthEvent, &listener, _1));

    // run for 2 secs of simulated events
    auto result = driver.Run(slammer::Timediff(60.0));
    EXPECT_TRUE(result.ok());
    EXPECT_LE(listener.max_time - listener.min_time, Timediff(10.0));

    // 2 secs @ 30 frames/sec, minus 1 frame because the recording doesn't start with a frame at the very beginning
    EXPECT_EQ(listener.num_frames, 59);

    // So we stepped through the code convincing ourselves that this result is meaningful
    EXPECT_EQ(listener.num_keyframes, 4);

    // TODO: Compare estimated distance (which determines the number of keyframes) against provided ground truth
    // and/or the provided odometer readings. This may not be the same, but should be somewhat close
}