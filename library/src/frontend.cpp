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
#include "slammer/math.h"
#include "slammer/backend.h"


using namespace slammer;

namespace {

template <typename Data, typename Mask>
size_t Compress(Data& data, const Mask& mask) {
    assert(data.size() == mask.size());

    using std::begin, std::end;
    size_t num_elements = 0;

    for (size_t index = 0; index != mask.size(); ++index) {
        if (mask[index]) {
            data[num_elements] = data[index];
            ++num_elements;
        }
    }

    data.resize(num_elements);
    return num_elements;
}

template <typename Data, typename Mask>
size_t CompressInto(const Data& input, const Mask& mask, Data& output) {
    assert(input.size() == mask.size());
    
    using std::begin, std::end;
    size_t num_elements = 0;
    output.clear();

    for (size_t index = 0; index != mask.size(); ++index) {
        if (mask[index]) {
            output.push_back(input[index]);
            ++num_elements;
        }
    }

    return num_elements;
}

}

RgbdFrontend::RgbdFrontend(const Parameters& parameters, const Camera& rgb_camera, const Camera& depth_camera)
    : parameters_(parameters), rgb_camera_(rgb_camera), depth_camera_(depth_camera), 
      status_(Status::kInitializing), random_engine_(parameters.seed) {
    feature_detector_ = 
        //cv::GFTTDetector::create(parameters.num_features, parameters.quality_level, parameters.min_distance);
        cv::ORB::create(parameters_.num_features);
}

void RgbdFrontend::ClearFeatureVectors() {
    tracked_features_.clear();
    tracked_feature_coords_.clear();
}

void RgbdFrontend::PostKeyframe() {
    last_keyframe_pose_ = current_pose_;
    
    RgbdFrameEvent event;

    last_keyframe_timestamp_ =
        event.timestamp = std::max(current_frame_data_.time_rgb, current_frame_data_.time_depth);
    event.frame_data = current_frame_data_;
    event.pose = current_pose_;
    event.keypoints = key_points_;
    event.info.rgb = &rgb_camera_;
    event.info.depth = &depth_camera_;

    feature_detector_->compute(current_frame_data_.rgb, event.keypoints, event.descriptions);

    keyframes.HandleEvent(event);
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

void RgbdFrontend::HandleKeyframePoseEvent(const KeyframePoseEvent& event) {
    // TODO: Implement this
    assert(false);
}

void RgbdFrontend::ProcessFrame() {
    // Haven't received enough information yet; skip processing
    if (current_frame_data_.rgb.empty() || current_frame_data_.depth.empty())
        return;

    Timestamp now = std::max(current_frame_data_.time_depth, current_frame_data_.time_rgb);

    // estimate the current pose 
    auto relative_motion = SE3d::exp(relative_motion_twist_ * (last_processed_time_ - now).count());
    current_pose_ = relative_motion * previous_pose_;

    switch (status_) {
    case Status::kInitializing:
        // For now the same as generation of new keyframe; may do more callibration in the future

    case Status::kNewKeyframe:
        {
            ClearFeatureVectors();
            size_t num_features = DetectKeyframeFeatures();

            // if we have enough features, we can attempt to track in the next frame
            if (num_features >= parameters_.num_features_tracking) {
                PostKeyframe();
                status_ = Status::kTracking;
            } else {
                // Are we sitting in the dark?
                // Standing in front of a blank wall?
                // What's reasonable recovery or behavior?
            }
        }
        break;

    case Status::kTracking:
        {
            previous_frame_data_ = current_frame_data_;
            std::vector<Point2f> current_points; 

            size_t num_features = FindFeaturesInCurrent(current_points);
            std::vector<Point3d> current_coords;

            for (const auto& point: current_points) {
                float row = point.x, column = point.y; 
                auto z = static_cast<double>(current_frame_data_.depth.at<ushort>(floorf(row), floorf(column)));
                current_coords.push_back(rgb_camera_.PixelToCamera(point, z));
            }    

            SE3d relative_motion;
            std::vector<uchar> mask;

            num_features = 
                RobustIcp(tracked_feature_coords_, current_coords,
                          random_engine_, relative_motion, mask,
                          parameters_.max_iterations, parameters_.sample_size, 
                          parameters_.outlier_factor);

            CompressInto(current_points, mask, tracked_features_);
            CompressInto(current_coords, mask, tracked_feature_coords_);

            num_features = tracked_features_.size();
            current_pose_ = relative_motion * previous_pose_;

            SE3d motion_since_last_keyframe = current_pose_ * last_keyframe_pose_.inverse();
            double traveled_distance = sqrt(motion_since_last_keyframe.translation().lpNorm<2>());

            if (traveled_distance >= parameters_.max_keyframe_distance ||
                num_features < parameters_.num_features_tracking_bad) {
                // need to create a new keyframe in next iteration
                status_ = Status::kNewKeyframe;
            } else {
                // attempt to improve tracking with additional features
                DetectAdditionalFeatures();
            }
        }

        break;
    }

    // TODO: This should be processed before a HandleKeyframePoseEvent gets handled
    previous_frame_data_ = current_frame_data_;
    relative_motion_ = current_pose_ * previous_pose_.inverse();
    relative_motion_twist_ = relative_motion_.log() * (1.0/(last_processed_time_ - now).count());
    previous_pose_ = current_pose_;
    last_processed_time_ = now;
}

size_t RgbdFrontend::DetectKeyframeFeatures() {
    using std::begin, std::end;

    feature_detector_->detect(current_frame_data_.rgb, key_points_);

    // add new features to tracked feature set
    cv::KeyPoint::convert(key_points_, tracked_features_);

    for (const auto& point: key_points_) {
        float row = point.pt.x, column = point.pt.y; 
        auto z = static_cast<double>(current_frame_data_.depth.at<ushort>(floorf(row), floorf(column)));
        tracked_feature_coords_.push_back(rgb_camera_.PixelToCamera(point.pt, z));
    }    

    return tracked_features_.size();
}

size_t RgbdFrontend::DetectAdditionalFeatures() {
    using std::begin, std::end;

    Point2f offset(parameters_.feature_mask_size, parameters_.feature_mask_size);
    cv::Mat feature_mask(current_frame_data_.rgb.size(), CV_8UC1, 255);

    for (const auto &point: tracked_features_) {
        cv::rectangle(feature_mask, point - offset, point + offset, 0, cv::FILLED);
    }

    std::vector<cv::KeyPoint> additional_points;
    feature_detector_->detect(current_frame_data_.rgb, additional_points, feature_mask);

    for (const auto& point: additional_points) {
        float row = point.pt.x, column = point.pt.y; 
        auto z = static_cast<double>(current_frame_data_.depth.at<ushort>(floorf(row), floorf(column)));
        tracked_features_.push_back(point.pt);
        tracked_feature_coords_.push_back(rgb_camera_.PixelToCamera(point.pt, z));
    }    

    return tracked_features_.size();
}

size_t RgbdFrontend::FindFeaturesInCurrent(std::vector<Point2f>& current_points) {
    // We initialize key point locations in the current frame with their location in the previous frame.
    // In a later iteration, we may want to predict the new position using additional information, such
    // as by assuming linear motion in pixel space (using flow directions calculated in a previous iteration)
    // or by backprojecting the 3d position using an estimated camera pose for the current frame.
    // Initialization of the current point locations requires setting the `cv::OPTFLOW_USE_INITIAL_FLOW flag`
    // in `cv::calcOpticalFlowPyrLK`.
    if (status_ == Status::kTracking) {
        PredictFeaturesInCurrent(current_pose_, current_points);
    } else {
        current_points = tracked_features_;
    }

    cv::Mat error;
    cv::TermCriteria termination_criteria {
        cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 
        parameters_.flow_iterations_max, 
        parameters_.flow_threshold
    };

    cv::Size window(parameters_.flow_window_size, parameters_.flow_window_size);
    std::vector<uchar> mask;

    cv::calcOpticalFlowPyrLK(previous_frame_data_.rgb, current_frame_data_.rgb, 
                             tracked_features_, current_points, 
                             mask, error, window, parameters_.flow_pyramid_levels,
                             termination_criteria, cv::OPTFLOW_USE_INITIAL_FLOW);

    Compress(tracked_features_, mask);
    Compress(tracked_feature_coords_, mask);
    Compress(current_points, mask);

    return current_points.size();
}

void RgbdFrontend::PredictFeaturesInCurrent(const SE3d& predicted_pose, std::vector<Point2f>& points) {
    points.clear();

    auto camera_to_robot = rgb_camera_.camera_to_robot();
    auto transform = camera_to_robot.inverse() * predicted_pose.inverse() * previous_pose_ * camera_to_robot;

    for (const auto& point: tracked_features_) {
        float row = point.x, column = point.y; 
        // Question: Just use value from single pixel, or do a weighted average across a small neighborhood?
        // What is the right rounding mode that's compatible with OpenCV feature and flow calculations?
        auto z = static_cast<double>(current_frame_data_.depth.at<ushort>(floorf(row), floorf(column)));
        // Question: Which is the right form to do this?
        // Point3d image_coord(column * z, row * z, z);
        auto transformed_point = transform * rgb_camera_.PixelToCamera(point, z);
        points.push_back(rgb_camera_.CameraToPixel(transformed_point));
    }
}

/*

changes to the tracking algorithm:

- Keypoints are calculated on an initial frame
- If those key points are tracked successfully into the following frame, the previous frame and the trackable features
    are submitted as key frame to the backend
- For the purpose of performing VO, only 3d coordinates are utilized further (for predicting the location of features
    and for estimating the relative motion)
- The set of points utilized is reduced frame over frame due to the inability to match them or because they end up
    as outliers during pose estimation
- Determine if a mask or an index vector is more effective for tracking subsetting of the point set
- the relative motion is stored and can be applied to extrapolate a pose estimate (check Sophus interpolate function)
- restart looking for a new key frame once we have lost sufficiently many features in this process, we have traveled a
    certain distance (parameter) or if we have exceeded a certain time interval (parameter)

- IMU will inform motion estimates if/once available (versus extrapolation)

*/