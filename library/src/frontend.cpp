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


using namespace slammer;

namespace {

// Depth scale used: https://lifelong-robotic-vision.github.io/dataset/scene
static constexpr double kDepthScale = 0.001;

inline double DepthForPixel(const DepthImage& depth_image, Point2f point) {
    float column = point.x, row = point.y; 
    auto depth = const_view(*depth_image).at(floorf(column), floorf(row))[0];

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

} // namespace 

void AbsoluteTracker::HandleColorEvent(const ColorImageEvent& event) {
    current_state_.color_image = event.image;

    if (parameters_.trigger == Trigger::kTriggerColor) {
        ProcessCurrentImages(event.timestamp);
    }
}

void AbsoluteTracker::HandleDepthEvent(const DepthImageEvent& event) {
    current_state_.depth_image = event.image;

    if (parameters_.trigger == Trigger::kTriggerDepth) {
        ProcessCurrentImages(event.timestamp);
    }
}

void AbsoluteTracker::ProcessCurrentImages(Timestamp timestamp) {
    // TODO: Later, we could actually learn if color or depth should be used as trigger,
    // or possibly use some form of interpolation

    // Let's make sure we have both images required
    if (!current_state_.color_image || !current_state_.depth_image) {
        return;
    }

    if (skip_count_) {
        --skip_count_;
        return;
    }

    skip_count_ = parameters_.skip_count;

    // Dispatch to correct state function
    switch (state_) {
    case TrackingState::kInitializing:
        DoInitialize(timestamp);
        break;

    case TrackingState::kTracking:
    case TrackingState::kTrackingBad:
        DoTracking(timestamp);
        break;

    case TrackingState::kTrackingLost:
        DoRecover(timestamp);
        break;
    }
}

void AbsoluteTracker::DidInitialize() {
    if (initialization.has_listeners()) {
        InitializedEvent event {
            reference_state_.timestamp,
            reference_state_.pose,
            reference_state_.pixel_coords.size(),
            reference_state_.color_image,
            reference_state_.pixel_coords
        };

        initialization.HandleEvent(event);
    }
}

bool AbsoluteTracker::ReferenceFromCurrent() {
    std::vector<orb::KeyPoint> key_points;
    Descriptors descriptors;

    detector_.ComputeFeatures(const_view(*current_state_.color_image), 
                                parameters_.tracking.num_features, key_points,
                                &descriptors);

    if (key_points.size() >= parameters_.tracking.num_features_init) {
        // capture the images as reference
        reference_state_.color_image = current_state_.color_image;
        reference_state_.depth_image = current_state_.depth_image;

        // add new features to tracked feature set
        reference_state_.pixel_coords.clear();
        reference_state_.camera_coords.clear();

        for (const auto& point: key_points) {
            auto z = DepthForPixel(current_state_.depth_image, point.coords);

            if (!isnan(z)) {
                reference_state_.pixel_coords.push_back(point.coords);
                auto camera_point = rgb_camera_.PixelToCamera(point.coords, z);
                reference_state_.camera_coords.push_back(camera_point);
            }
        }    

        reference_state_.timestamp = current_state_.timestamp;
        reference_state_.pose = current_state_.pose;

        if (key_points.size() >= parameters_.tracking.num_features_needed_for_keyframe) {
            PostKeyframe(reference_state_, descriptors);
        }

        return true;
    } else {
        return false;
    }
}

void AbsoluteTracker::DoInitialize(Timestamp timestamp) {
    // keep track of time
    current_state_.timestamp = timestamp;

    // initialize the current and reference pose to origin; this may need to changed later
    current_state_.pose = SE3d();

    // initialize the "velocity" to zero
    relative_motion_twist_.setZero();

    if (ReferenceFromCurrent()) {
        // update state 
        state_ = TrackingState::kTracking;

        DidInitialize();       
    }
}

void AbsoluteTracker::DidTrack() {
    if (tracking.has_listeners()) {
        TrackingEvent event {
            current_state_.timestamp,
            current_state_.pose,
            reference_state_.pixel_coords.size(),
            current_state_.color_image,
            current_state_.pixel_coords,
            reference_state_.pixel_coords
        };

        tracking.HandleEvent(event);
    }
}

void AbsoluteTracker::LostTracking() {
    if (tracking_lost.has_listeners()) {
        TrackingLostEvent event {
            current_state_.timestamp,
        };

        tracking_lost.HandleEvent(event);
    }
}

void AbsoluteTracker::DoTracking(Timestamp timestamp) {
    std::vector<float> error;

    // TODO: Refactor this such that the same conversion can be used for feature detection
    auto source = RgbToGrayscale(const_view(*reference_state_.color_image));
    auto target = RgbToGrayscale(const_view(*current_state_.color_image));

    // last tracked pose versus refrence pose
    auto incremental_pose = reference_state_.pose.inverse() * current_state_.pose;

    auto arg = relative_motion_twist_ * (timestamp - current_state_.timestamp).count();
    auto estimated_motion = SE3d::exp(arg);
    auto estimated_transform = (incremental_pose * estimated_motion).inverse();

    current_state_.pixel_coords.clear();
    std::transform(reference_state_.camera_coords.begin(), reference_state_.camera_coords.end(),
        std::back_insert_iterator(current_state_.pixel_coords),
        [&](const auto& point) { return rgb_camera_.CameraToPixel(estimated_transform * point); });

    ComputeFlow(const_view(source), const_view(target),
                reference_state_.pixel_coords, current_state_.pixel_coords, error,
                parameters_.flow.pyramid_levels, parameters_.flow.omega, parameters_.flow.threshold,
                parameters_.flow.max_iterations);

    float squared_error = parameters_.flow.threshold * parameters_.flow.threshold;

    // this mask captures the points we have been able to track well
    // that is, they are still within the valid range of coordinates, and the error is controlled
    std::vector<uchar> mask;
    for (size_t index = 0; index < current_state_.pixel_coords.size(); ++index) {
        const auto& point = current_state_.pixel_coords[index];
        if (!isnan(point.x) && !isnan(point.y) && error[index] <= squared_error) {
            mask.push_back(std::numeric_limits<uchar>::max());
        } else {
            mask.push_back(0);
        }
    }

    // create 3d coordinates for tracked features
    current_state_.camera_coords.clear();

    for (size_t index = 0; index < current_state_.pixel_coords.size(); ++index) {
        if (!mask[index]) {
            current_state_.camera_coords.push_back(Point3d());
            continue;
        }

        const auto& point = current_state_.pixel_coords[index];
        auto z = DepthForPixel(current_state_.depth_image, point);

        if (isnan(z)) {
            mask[index] = 0;
        }

        auto camera_point = rgb_camera_.PixelToCamera(point, z);
        current_state_.camera_coords.push_back(camera_point);
    }   

    PerspectiveAndPoint3d::PointPairs point_pairs;

    for (size_t index = 0; index < current_state_.pixel_coords.size(); ++index) {
        point_pairs.emplace_back(depth_camera_.CameraToPixelDisparity(reference_state_.camera_coords[index]), 
                                    depth_camera_.CameraToPixelDisparity(current_state_.camera_coords[index]));
    }

    PerspectiveAndPoint3d instance(depth_camera_, depth_camera_, point_pairs);
    SE3d calculated_pose = incremental_pose.inverse();//estimated_transform;

    auto result =
        instance.Ransac(calculated_pose, mask, reference_state_.camera_coords, 
                        parameters_.pnp.sample_size, parameters_.pnp.max_iterations, 
                        parameters_.pnp.lambda, parameters_.pnp.outlier_threshold, 
                        random_engine_);

    if (!result.ok()) {
        state_ = TrackingState::kTrackingLost;
        LostTracking();
        return;
    }

    auto new_pose = reference_state_.pose * calculated_pose.inverse();
    relative_motion_twist_ = 
        -(current_state_.pose * new_pose.inverse()).log() * (1.0/(timestamp - current_state_.timestamp).count());
    current_state_.pose = new_pose;
    current_state_.timestamp = timestamp;

    Compress(current_state_.pixel_coords, mask);
    Compress(reference_state_.pixel_coords, mask);
    Compress(reference_state_.camera_coords, mask);

    if (reference_state_.pixel_coords.size() < parameters_.tracking.num_features_tracking_bad) {
        state_ = TrackingState::kTrackingLost;
        LostTracking();
        return;
    }

    DidTrack();

    // Should we reset the reference?
    if (current_state_.timestamp - reference_state_.timestamp >= parameters_.tracking.max_duration ||
        calculated_pose.translation().norm() >= parameters_.tracking.max_distance) {
        // create a new reference state from the current state, and possibly post this as keyframe
        if (!ReferenceFromCurrent()) {
            state_ = TrackingState::kTrackingBad;
        }
    }
}

void AbsoluteTracker::DoRecover(Timestamp timestamp) {

}

void AbsoluteTracker::PostKeyframe(const TrackerState& state, const Descriptors& descriptors) {
    if (keyframes.has_listeners()) {
        keyframes.HandleEvent(CreateKeyFrameEvent(state, descriptors));
    }
}

KeyframeEvent AbsoluteTracker::CreateKeyFrameEvent(const TrackerState& state, const Descriptors& descriptors) {
    return KeyframeEvent {
        state.timestamp,
        state.pose,
        state.pixel_coords,
        state.camera_coords,
        descriptors,
        state.color_image,
        state.depth_image,
        rgb_camera_,
        depth_camera_
    };
}
