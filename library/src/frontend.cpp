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

#include "slammer/frontend.h"

#include "slammer/events.h"

using namespace slammer;

RgbdFrontend::RgbdFrontend(const Parameters& parameters, const Camera& rgb_camera, const Camera& depth_camera)
    : parameters_(parameters), rgb_camera_(rgb_camera), depth_camera_(depth_camera), status_(Status::kInitializing) {
    feature_detector_ = 
        cv::GFTTDetector::create(parameters.num_features, parameters.quality_level, parameters.min_distance);
}

void RgbdFrontend::HandleColorEvent(const ImageEvent& event) {
    current_frame_data_.time_rgb = event.timestamp;
    current_frame_data_.rgb = event.image;

    if (trigger_ == Trigger::kTriggerColor) {
        ProcessFrame();
    }
}

void RgbdFrontend::HandleDepthEvent(const ImageEvent& event) {
    current_frame_data_.time_depth = event.timestamp;
    current_frame_data_.depth = event.image;

    if (trigger_ == Trigger::kTriggerDepth) {
        ProcessFrame();
    }
}

void RgbdFrontend::ProcessFrame() {
    // Haven't received enough information yet; skip processing
    if (current_frame_data_.rgb.empty() || current_frame_data_.depth.empty())
        return;

    switch (status_) {
    case Status::kInitializing:
        // Just remember the image and return
        previous_frame_data_ = current_frame_data_;
        DetectFeatures(previous_frame_data_, previous_key_points_);

        // if we have enough features, we can attempt to track in the next frame
        if (previous_key_points_.size() >= parameters_.num_features_init) {
            status_ = Status::kTracking;
        }

        return;

    case Status::kTracking:

        previous_frame_data_ = current_frame_data_;
        break;

    case Status::kTrackingLost:
        break;
    }

    // calculate a new pose estimate
    //  - use a feature detector to identify feature points
    //  - use optical flow to determine offsets relative to previous image
    //  - apply RANSAC to calculate relative pose
    //  - update overall pose information
}

void RgbdFrontend::DetectFeatures(const RgbdFrameData& frame_data, KeyPoints& key_points) {
    using std::begin, std::end;

    Point2f offset(parameters_.feature_mask_size, parameters_.feature_mask_size);
    cv::Mat feature_mask(frame_data.rgb.size(), CV_8UC1, 255);

    for (const auto &point: key_points) {
        cv::rectangle(feature_mask, point.pt - offset, point.pt + offset, 0, cv::FILLED);
    }

    std::vector<cv::KeyPoint> additional_points;
    feature_detector_->detect(frame_data.rgb, additional_points, feature_mask);

    key_points.insert(end(key_points), begin(additional_points), end(additional_points));
}

void RgbdFrontend::FindFeatureInCurrent(std::vector<Point2f>& points, std::vector<unsigned char>& mask) {
    std::vector<Point2f> previous_points;
    cv::KeyPoint::convert(previous_key_points_, previous_points);

    // We initialize key point locations in the current frame with their location in the previous frame.
    // In a later iteration, we may want to predict the new position using additional information, such
    // as by assuming linear motion in pixel space (using flow directions calculated in a previous iteration)
    // or by backprojecting the 3d position using an estimated camera pose for the current frame.
    std::vector<Point2f> current_points { previous_points };

    cv::Mat error;
    cv::TermCriteria termination_criteria {
        cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 
        parameters_.flow_iterations_max, 
        parameters_.flow_threshold
    };

    cv::Size window(parameters_.flow_window_size, parameters_.flow_window_size);

    cv::calcOpticalFlowPyrLK(previous_frame_data_.rgb, current_frame_data_.rgb, 
                             previous_points, current_points, 
                             mask, error, window, parameters_.flow_pyramid_levels,
                             termination_criteria, cv::OPTFLOW_USE_INITIAL_FLOW);
}
