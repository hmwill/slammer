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

#include "slammer/slammer.h"

#include "slammer/pipeline.h"
#include "slammer/loris/driver.h"
#include "slammer/loris/opencv_utils.h"

#include "opencv2/opencv.hpp"
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/viz.hpp"

using namespace slammer;
using namespace slammer::loris;

namespace {

std::vector<cv::Vec3d> EigenToOpenCV(const std::vector<Point3d>& eigen) {
    std::vector<cv::Vec3d> result;

    std::transform(eigen.begin(), eigen.end(), std::back_inserter(result),
                   [](const Point3d& point) { return cv::Vec3d(point.x(), point.y(), point.z()); });

    return result;
}

}

void RunPipeline(Timediff duration) {
    using namespace std::placeholders;

    std::string kDataSetPath("data/cafe1-1");

    Result<SensorInfo> sensor_info_result = ReadSensorInfo(kDataSetPath);
    assert(sensor_info_result.ok());
    auto sensor_info = sensor_info_result.value();

    Result<FrameSet> frame_info_result = ReadFrames(kDataSetPath);
    assert(frame_info_result.ok());
    auto frame_info = frame_info_result.value();

    Driver driver(kDataSetPath, arrow::io::default_io_context());

    auto rgb_camera_pose = GetFramePose(frame_info, "d400_color_optical_frame");
    assert(rgb_camera_pose.ok());
    auto depth_camera_pose = GetFramePose(frame_info, "d400_depth_optical_frame");
    assert(depth_camera_pose.ok());

    Camera rgb_camera = CreateCamera(sensor_info.d400_color_optical_frame, rgb_camera_pose.value());
    Camera depth_camera = CreateCamera(sensor_info.d400_depth_optical_frame, depth_camera_pose.value());
    RgbdFrontend::Parameters frontend_parameters;

    // process only 1 frane/sec
    frontend_parameters.skip_count = 1;
    frontend_parameters.max_keyframe_interval = Timediff(1.0);

    RgbdFrontend frontend(frontend_parameters, rgb_camera, depth_camera);

    driver.color.AddHandler(std::bind(&RgbdFrontend::HandleColorEvent, &frontend, _1));
    driver.aligned_depth.AddHandler(std::bind(&RgbdFrontend::HandleDepthEvent, &frontend, _1));

    auto listener = [&](const PointCloudAlignmentEvent& event) {
        cv::viz::WCloud reference_cloud(EigenToOpenCV(event.reference), cv::viz::Color::blue());
        reference_cloud.setRenderingProperty(cv::viz::POINT_SIZE, 3.);

        cv::viz::WCloud transformed_cloud(EigenToOpenCV(event.transformed), cv::viz::Color::red());
        transformed_cloud.setRenderingProperty(cv::viz::POINT_SIZE, 3.);

        cv::viz::Viz3d visualizer("Viz Window");
        visualizer.setBackgroundColor(cv::viz::Color::white());

        visualizer.showWidget("Reference", reference_cloud);
        visualizer.showWidget("Transformed", transformed_cloud);

        while (!visualizer.wasStopped()) {
            visualizer.spinOnce(/*1, true*/);
        }
    };

    frontend.point_cloud_alignments.AddHandler(listener);

    driver.Run(duration);
}

int main(int argc, char* argv[]) {
    RunPipeline(slammer::Timediff(10.0));
    return EXIT_SUCCESS;
}