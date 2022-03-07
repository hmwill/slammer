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
#include "slammer/flow.h"
#include "slammer/pnp.h"
#include "slammer/backend.h"


using namespace slammer;

namespace {

// Depth scale used: https://lifelong-robotic-vision.github.io/dataset/scene
static constexpr double kDepthScale = 0.001;

inline double DepthForPixel(const RgbdFrameData& frame, Point2f point) {
    float column = point.x, row = point.y; 
    auto depth = const_view(*frame.depth).at(floorf(column), floorf(row))[0];

    // 0 depth represents a missing depth value, so we convert those to an NaN value instead
    return depth ? kDepthScale * depth : std::numeric_limits<double>::signaling_NaN();
}

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

void MaskNaN(const std::vector<Point3d>& points, std::vector<uchar>& mask) {
    for (size_t index = 0; index < points.size(); ++index) {
        const auto& point = points[index];

        if (isnan(point.x() + point.y() + point.z())) {
            mask[index] = 0;
        } 
    }
}

inline bool NotNaN(const Point3d& point) {
    return !isnan(point.sum());
}

inline bool NotNaN(const std::vector<Point3d>& points) {
    return NotNaN(std::reduce(begin(points), end(points)));
}

}

RgbdFrontend::RgbdFrontend(const Parameters& parameters, const Camera& rgb_camera, const StereoDepthCamera& depth_camera)
    : parameters_(parameters), rgb_camera_(rgb_camera), depth_camera_(depth_camera), 
      feature_detector_(parameters.orb_parameters),
      status_(Status::kInitializing), random_engine_(parameters.seed), trigger_(Trigger::kTriggerColor),
      distance_since_last_keyframe_(0.0), skip_count_(0) {}

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
    event.descriptions = std::move(descriptors_);

    keyframes.HandleEvent(event);

    distance_since_last_keyframe_ = 0.0;
}

void RgbdFrontend::HandleColorEvent(const ColorImageEvent& event) {
    current_frame_data_.time_rgb = event.timestamp;
    current_frame_data_.rgb = event.image;

    if (trigger_ == Trigger::kTriggerColor) {
        ProcessFrame();
    }
}

void RgbdFrontend::HandleDepthEvent(const DepthImageEvent& event) {
    current_frame_data_.time_depth = event.timestamp;
    current_frame_data_.depth = event.image;

    if (trigger_ == Trigger::kTriggerDepth) {
        ProcessFrame();
    }
}

void RgbdFrontend::HandleKeyframePoseEvent(const KeyframePoseEvent& event) {
    assert(event.timestamp == event.keyframe->timestamp);
    keyframe_pose_updates_.push(KeyframePoseUpdate {
        event.timestamp,
        event.previous_pose,
        event.keyframe->pose
    });
}

void RgbdFrontend::ProcessFrame() {
    // Haven't received enough information yet; skip processing
    if (!current_frame_data_.rgb || !current_frame_data_.depth)
        return;

    if (skip_count_) {
        --skip_count_;
        return;
    }

    skip_count_ = parameters_.skip_count;

    Timestamp now = std::max(current_frame_data_.time_depth, current_frame_data_.time_rgb);

    // process any pending keyframe pose updates that the backend may have sent
    while (!keyframe_pose_updates_.empty()) {
        auto pose_update = keyframe_pose_updates_.front();
        keyframe_pose_updates_.pop();

        if (pose_update.timestamp <= last_keyframe_timestamp_) {

            // relative_motion_ stays as is
            // relative_motion_twist_stays as is
            
            auto last_relative = previous_pose_ * last_keyframe_pose_.inverse();
            last_keyframe_pose_ = last_keyframe_pose_ * pose_update.previous_pose.inverse() * pose_update.new_pose;
            previous_pose_ = last_relative * last_keyframe_pose_;
        }
    }

    Status old_status = status_;
    bool is_keyframe = false;

    switch (status_) {
    case Status::kInitializing:
        // For now the same as generation of new keyframe; may do more callibration in the future

    case Status::kNewKeyframe:
        {
            // estimate the current pose; this could be any pose, actually, because we are starting over
            current_pose_ = previous_pose_;

            ClearFeatureVectors();
            size_t num_features = DetectKeyframeFeatures();

            // if we have enough features, we can attempt to track in the next frame
            if (num_features >= parameters_.num_features_tracking) {
                PostKeyframe();
                distance_since_last_keyframe_ = 0;
                is_keyframe = true;
                status_ = Status::kTracking;
            } else {
                assert(false);
                abort();
            }
        }
        break;

    case Status::kTracking:
        {
            // estimate the current pose 
            auto relative_motion = SE3d::exp(relative_motion_twist_ * (last_processed_time_ - now).count());
            current_pose_ = relative_motion * previous_pose_;

            previous_frame_data_ = current_frame_data_;
            std::vector<Point2f> current_points; 

            size_t num_features = FindFeaturesInCurrent(current_points);
            std::vector<Point3d> current_coords;

            // TODO: num_features < parameters_.num_features_tracking_bad
            std::vector<uchar> mask;

            for (const auto& point: current_points) {
                auto z = DepthForPixel(current_frame_data_, point);
                current_coords.push_back(rgb_camera_.PixelToCamera(point, z));
                mask.push_back(!isnan(z) ? std::numeric_limits<uchar>::max() : 0);
            }    

            //assert(NotNaN(tracked_feature_coords_));
            //assert(NotNaN(current_coords));

            PerspectiveAndPoint3d::PointPairs point_pairs;

            for (size_t index = 0; index < tracked_feature_coords_.size(); ++index) {
                point_pairs.emplace_back(depth_camera_.CameraToPixelDisparity(tracked_feature_coords_[index]), 
                                         depth_camera_.CameraToPixelDisparity(current_coords[index]));
            }

            PerspectiveAndPoint3d instance(depth_camera_, depth_camera_, point_pairs);
            SE3d calculated_pose = current_pose_.inverse();

            auto result =
                instance.Ransac(calculated_pose, mask, tracked_feature_coords_, parameters_.sample_size, parameters_.max_iterations, parameters_.lambda, parameters_.outlier_threshold, random_engine_);

            // TODO: What happens if we could not solve the optimization problem?
            assert(result.ok());

            relative_motion = calculated_pose.inverse();
            PostPointCloudAlignment(now, relative_motion, tracked_feature_coords_, current_coords, mask);
            
            CompressInto(current_points, mask, tracked_features_);
            CompressInto(current_coords, mask, tracked_feature_coords_);

            num_features = tracked_features_.size();
            current_pose_ = relative_motion * previous_pose_;

            SE3d motion_since_last_keyframe = current_pose_ * last_keyframe_pose_.inverse();
            double distance_since_last_keyframe_ = motion_since_last_keyframe.translation().norm();

            if (distance_since_last_keyframe_ >= parameters_.max_keyframe_distance ||
                num_features < parameters_.num_features_tracking_bad ||
                now - last_keyframe_timestamp_ > parameters_.max_keyframe_interval) {
                // need to create a new keyframe in next iteration
                status_ = Status::kNewKeyframe;
            } else {
                // attempt to improve tracking with additional features
                if (num_features < parameters_.num_features_tracking) {
                    DetectAdditionalFeatures(parameters_.num_features - num_features);
                }
            }
        }

        break;
    }

    previous_frame_data_ = current_frame_data_;
    relative_motion_ = current_pose_ * previous_pose_.inverse();
    relative_motion_twist_ = relative_motion_.log() * (1.0/(last_processed_time_ - now).count());
    previous_pose_ = current_pose_;
    last_processed_time_ = now;

    PostProcessedFrame(now, old_status, status_, current_pose_, tracked_features_.size(), is_keyframe);
}

size_t RgbdFrontend::DetectKeyframeFeatures() {
    using std::begin, std::end;

    feature_detector_.ComputeFeatures(const_view(*current_frame_data_.rgb), 
                                        parameters_.num_features, key_points_, &descriptors_);

    // add new features to tracked feature set
    tracked_features_.clear();
    tracked_feature_coords_.clear();

    for (const auto& point: key_points_) {
        auto z = DepthForPixel(current_frame_data_, point.coords);

        if (!isnan(z)) {
            tracked_features_.push_back(point.coords);
            auto camera_point = rgb_camera_.PixelToCamera(point.coords, z);
            assert(NotNaN(camera_point));
            tracked_feature_coords_.push_back(camera_point);
        }
    }    

    return tracked_features_.size();
}

size_t RgbdFrontend::DetectAdditionalFeatures(size_t num_additonal) {
    using std::begin, std::end;
    using namespace boost::gil;

    gray8_image_t feature_mask(current_frame_data_.rgb->width(), current_frame_data_.rgb->height());
    auto mask = view(feature_mask); 
    const auto kFillValue = gray8_pixel_t(std::numeric_limits<gray8_pixel_t::value_type>::max());

    for (const auto &point: tracked_features_) {
        int x_min = std::max(ptrdiff_t(floorf(point.x)) - parameters_.orb_parameters.min_distance, (ptrdiff_t) 0);
        int x_max = std::min(ptrdiff_t(floorf(point.x)) + parameters_.orb_parameters.min_distance + 1, mask.width());
        int y_min = std::max(ptrdiff_t(floorf(point.y)) - parameters_.orb_parameters.min_distance, (ptrdiff_t) 0);
        int y_max = std::min(ptrdiff_t(floorf(point.y)) + parameters_.orb_parameters.min_distance + 1, mask.height());

        for (auto y = y_min; y < y_max; ++y) {
            auto iter = mask.row_begin(y);

            for (auto x = x_min; x < x_max; ++x) {
                iter[x] = kFillValue;
            }
        }
    }

    KeyPoints additional_points;
    auto const_mask = const_view(feature_mask);

    feature_detector_.ComputeFeatures(const_view(*current_frame_data_.rgb), 
                                      parameters_.num_features, additional_points, nullptr, &const_mask);
        
    if (additional_points.size() > num_additonal) {
        additional_points.resize(num_additonal);
    }

    for (const auto& point: additional_points) {
        auto z = DepthForPixel(current_frame_data_, point.coords);

        if (!isnan(z)) {
            tracked_features_.push_back(point.coords);
            tracked_feature_coords_.push_back(rgb_camera_.PixelToCamera(point.coords, z));
        }
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

    std::vector<float> error;

    // TODO: Refactor this such that the same conversion can be used for feature detection
    auto source = RgbToGrayscale(const_view(*previous_frame_data_.rgb));
    auto target = RgbToGrayscale(const_view(*current_frame_data_.rgb));

    ComputeFlow(const_view(source), const_view(target),
                tracked_features_, current_points, error,
                parameters_.flow_pyramid_levels, parameters_.flow_omega, parameters_.flow_threshold);

    float squared_error = parameters_.flow_threshold * parameters_.flow_threshold;

    std::vector<uchar> mask;
    std::transform(error.begin(), error.end(), std::back_inserter(mask),
                   [=](float err) { return err <= squared_error; });

    Compress(tracked_features_, mask);
    Compress(tracked_feature_coords_, mask);
    Compress(current_points, mask);

    return current_points.size();
}

void RgbdFrontend::PredictFeaturesInCurrent(const SE3d& predicted_pose, std::vector<Point2f>& points) {
    points.clear();

    auto camera_to_robot = rgb_camera_.camera_to_robot();
    //auto transform = camera_to_robot.inverse() * predicted_pose.inverse() * previous_pose_ * camera_to_robot;
    auto transform = predicted_pose * previous_pose_.inverse();

    for (const auto& point: tracked_features_) {
        auto z = DepthForPixel(current_frame_data_, point);
        auto transformed_point = transform * rgb_camera_.PixelToCamera(point, z);
        points.push_back(rgb_camera_.CameraToPixel(transformed_point));
    }
}


void RgbdFrontend::PostProcessedFrame(Timestamp timestamp, Status old_state, Status new_state, const SE3d& pose,
                                      size_t num_tracked_features, bool is_keyframe) {
    if (processed_frames.has_listeners()) {
        ProcessedFrameEvent event;

        event.timestamp = timestamp;
        event.old_state = old_state;
        event.new_state = new_state;
        event.pose = pose;
        event.num_tracked_features = num_tracked_features;
        event.is_keyframe = is_keyframe;

        processed_frames.HandleEvent(event);
    }
}

void 
RgbdFrontend::PostPointCloudAlignment(Timestamp timestamp, const SE3d& relative_motion,
                                      const std::vector<Point3d>& reference, 
                                      const std::vector<Point3d>& transformed,
                                      const std::vector<uchar>& inliers) {
    if (point_cloud_alignments.has_listeners()) {
        PointCloudAlignmentEvent event;

        event.timestamp = timestamp;
        event.relative_motion = relative_motion;
        event.reference = reference;
        event.transformed = transformed;
        std::transform(inliers.begin(), inliers.end(), std::back_inserter(event.inliers),
                       [](uchar ch) { return ch != 0; });

        point_cloud_alignments.HandleEvent(event);
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

