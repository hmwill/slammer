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
#include <sciplot/sciplot.hpp>

#include "slammer/slammer.h"
#include "slammer/frontend.h"
#include "slammer/flow.h"
#include "slammer/pnp.h"

#include "slammer/loris/driver.h"
#include "slammer/loris/opencv_utils.h"


using namespace slammer;
using namespace slammer::loris;

namespace {

using namespace boost::gil;

void LogFeatures(ImageLogger& logger, const std::string& name, const gray8c_view_t& image,
                 const std::vector<Point2f>& points, const std::vector<Point2f>* secondary = nullptr) {
    rgb8_image_t display_features(image.dimensions());
    auto output = view(display_features);

    copy_pixels(color_converted_view<rgb8_pixel_t>(image), output);

    const rgb8_pixel_t red(255, 0, 0);
    const rgb8_pixel_t blue(0, 0, 255);

    if (secondary) {
        for (auto point: *secondary) {
            output(point.x, point.y) = blue;
            output(point.x - 3, point.y) = blue;
            output(point.x - 2, point.y) = blue;
            output(point.x - 1, point.y) = blue;
            output(point.x + 1, point.y) = blue;
            output(point.x + 2, point.y) = blue;
            output(point.x + 3, point.y) = blue;
            output(point.x, point.y - 3) = blue;
            output(point.x, point.y - 2) = blue;
            output(point.x, point.y - 1) = blue;
            output(point.x, point.y + 1) = blue;
            output(point.x, point.y + 2) = blue;
            output(point.x, point.y + 3) = blue;
        }
    }

    for (auto point: points) {
        if (isnan(point.x) || isnan(point.y)) {
            continue;
        }
        
        output(point.x, point.y) = red;
        output(point.x - 3, point.y) = red;
        output(point.x - 2, point.y) = red;
        output(point.x - 1, point.y) = red;
        output(point.x + 1, point.y) = red;
        output(point.x + 2, point.y) = red;
        output(point.x + 3, point.y) = red;
        output(point.x, point.y - 3) = red;
        output(point.x, point.y - 2) = red;
        output(point.x, point.y - 1) = red;
        output(point.x, point.y + 1) = red;
        output(point.x, point.y + 2) = red;
        output(point.x, point.y + 3) = red;
    }

    logger.LogImage(const_view(display_features), name);
}

class TrackingListener {
public:
    TrackingListener(const std::string& path)
      : image_logger_(path) {}

    void HandleGroundtruthEvent(const loris::GroundtruthEvent& event) {
        current_groundtruth_pose = SE3d(event.orientation, Point3d::Zero()) * 
            SE3d(SE3d::QuaternionType::Identity(), event.position);
        groundtruth_path.push_back(event.position);
    }

    void HandleInitializedEvent(const InitializedEvent& event) {
        ++num_frames;
        current_timestamp = event.timestamp;
        refernce_timestamp = event.timestamp;
        reference_pose = event.pose;
        reference_groundtruth_pose = current_groundtruth_pose;
        estimated_path.push_back(event.pose);

        auto gray = RgbToGrayscale(const_view(*event.image));
        LogFeatures(image_logger_, fmt::format("initialized-{}.", event.timestamp.time_since_epoch().count()), 
                    const_view(gray), event.features);
    }

    void HandleTrackingEvent(const TrackingEvent& event) {
        ++num_frames;
        is_tracking = true;
        current_timestamp = event.timestamp;
        current_pose = event.pose;
        estimated_path.push_back(event.pose);

        auto groundtruth_translation = (current_groundtruth_pose * reference_groundtruth_pose.inverse()).translation();
        auto groundtruth_distance = groundtruth_translation.norm();
        auto estimated_translation = (current_pose * reference_pose.inverse()).translation();
        auto estimated_distance = estimated_translation.norm();

        // should be within 90%?
        auto diff = estimated_distance - groundtruth_distance;
        //EXPECT_LE(fabs(diff), 0.1);

        auto factor = estimated_distance / groundtruth_distance;

        //EXPECT_GE(factor, 0.75);
        //EXPECT_LE(factor, 1.25);

        // EXPECT_EQ(estimated_distance, groundtruth_distance);

        auto gray = RgbToGrayscale(const_view(*event.image));
        LogFeatures(image_logger_, fmt::format("tracking-{}.", event.timestamp.time_since_epoch().count()), 
                    const_view(gray), event.features, &event.reference);
    }

    void HandleTrackingLostEvent(const TrackingLostEvent& event) {
        current_timestamp = event.timestamp;
        is_tracking = false;
    }

    void HandleKeyframeEvent(const KeyframeEvent& event) {
        ++num_keyframes;
        image_logger_.LogImage(const_view(*event.color), 
                               fmt::format("keyframe-{}.", event.timestamp.time_since_epoch().count()));
    }

    FileImageLogger image_logger_;

    size_t num_frames = 0;
    size_t num_keyframes = 0;
    bool is_tracking = false;

    SE3d current_groundtruth_pose;
    SE3d reference_groundtruth_pose;
    SE3d reference_pose;
    SE3d current_pose;
    Timestamp refernce_timestamp;
    Timestamp current_timestamp;
    std::vector<Point3d> groundtruth_path;
    std::vector<SE3d> estimated_path;
};

void PlotPath(const std::string& filename, const std::vector<Point3d>& locations, bool use_z = true) {
    using namespace sciplot;

    Plot plot;

    std::vector<double> x, y, z;

    for (const auto& location: locations) {
        x.push_back(location.x());
        y.push_back(location.y());
        z.push_back(location.z());
    }

    plot.drawCurve(x, use_z ? z : y);

    plot.save(filename);
}

void PlotPath(const std::string& filename, const std::vector<SE3d>& poses, bool use_z = true) {
    using namespace sciplot;

    Plot plot;

    std::vector<double> x, y, z;

    for (const auto& pose: poses) {
        const auto& translation = pose.translation();
        x.push_back(translation.x());
        y.push_back(translation.y());
        z.push_back(translation.z());
    }

    plot.drawCurve(x, use_z ? z : y);

    plot.save(filename);
}

} // namespace

// Test tracking relative to the first frame
//
// Debugging information to generate along the way:
//  - Image logs showing how features are mapped
//  - Pose relative to first frame as calculated
//  - Pose relative to first frame as provided by ground truth
//
// To consider: instead of tracking via optical flow, we could equally
// compare the results to key point detection and matchng for each frame
TEST(FrontendTest, TestFrontend) {
    using namespace std::placeholders;

    std::string kDataSetPath("data/cafe1-1");
    Driver driver(kDataSetPath);
    
    // get the color and depth camera objects for the data set
    Result<SensorInfo> sensor_info_result = ReadSensorInfo(kDataSetPath);
    EXPECT_TRUE(sensor_info_result.ok());
    auto sensor_info = sensor_info_result.value();

    Result<FrameSet> frame_info_result = ReadFrames(kDataSetPath);
    EXPECT_TRUE(frame_info_result.ok());
    auto frame_info = frame_info_result.value();

    auto rgb_camera_pose = GetFramePose(frame_info, "d400_color_optical_frame");
    EXPECT_TRUE(rgb_camera_pose.ok());
    auto depth_camera_pose = GetFramePose(frame_info, "d400_depth_optical_frame");
    EXPECT_TRUE(depth_camera_pose.ok());

    Camera rgb_camera = CreateCamera(sensor_info.d400_color_optical_frame, rgb_camera_pose.value());
    StereoDepthCamera depth_camera = CreateAlignedStereoDepthCamera(sensor_info.d400_depth_optical_frame, 0.05, depth_camera_pose.value());

    FrontendParameters parameters;

    parameters.tracking.max_duration = Timediff(0.9);

    Frontend tracker(parameters, rgb_camera, depth_camera);
    driver.color.AddHandler(std::bind(&Frontend::HandleColorEvent, &tracker, _1));
    driver.aligned_depth.AddHandler(std::bind(&Frontend::HandleDepthEvent, &tracker, _1));

    TrackingListener listener("image_logs/frontend_test/test_frontend");
    driver.groundtruth.AddHandler(std::bind(&TrackingListener::HandleGroundtruthEvent, &listener, _1));
    tracker.initialization.AddHandler(std::bind(&TrackingListener::HandleInitializedEvent, &listener, _1));
    tracker.tracking.AddHandler(std::bind(&TrackingListener::HandleTrackingEvent, &listener, _1));
    tracker.tracking_lost.AddHandler(std::bind(&TrackingListener::HandleTrackingLostEvent, &listener, _1));
    tracker.keyframes.AddHandler(std::bind(&TrackingListener::HandleKeyframeEvent, &listener, _1));

    // run for 5 secs of simulated events
    auto result = driver.Run(slammer::Timediff(5.0));
    EXPECT_TRUE(result.ok());
    EXPECT_TRUE(listener.is_tracking);
    EXPECT_EQ(listener.num_frames, 149);
    EXPECT_EQ(listener.num_keyframes, 6);

    PlotPath("image_logs/frontend_test/test_frontend/groundtruth.png", listener.groundtruth_path, false);
    PlotPath("image_logs/frontend_test/test_frontend/estimated.png", listener.estimated_path, true);
}

/*
- Render the sequence of poses; create such graphs both for the baseline and the estimated poses
- Add functionality to create new reference frames while keeping track of the current pose
- What criteria would trigger the creation of a new reference frame?
    - Low number of features tracked 
    - Degree to which the camera has swept the space
    - Are all features tracked confined to a small section of the image?
    - Number of frames/seconds since last reference frame was taken
    - Are there specific attributed that would make one frame preferable of others? (blurriness)
- How should we handle situations where tracking is lost; that is, how to restart and continue
- Should we handle situations where images cannot be captured? (e.g. lights off)
- Add feature descriptor calculation to reference frame processing
- Add posting of reference frames

- Refactor AbsoluteTracker into new FrontEnd class





*/