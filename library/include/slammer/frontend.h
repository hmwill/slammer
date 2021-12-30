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

#ifndef SLAMMER_RGBD_FRONTEND_H
#define SLAMMER_RGBD_FRONTEND_H

#pragma once

#include "slammer/slammer.h"

#include "opencv2/features2d.hpp"

#include "slammer/camera.h"
#include "slammer/events.h"

namespace slammer {

/// Representation of the camera data that is processed by the RGBD frontend. It comprises a color and a depth
/// image, each with an associated timestamp.
struct RgbdFrameData {
    cv::Mat rgb;
    cv::Mat depth;

    Timestamp time_rgb;
    Timestamp time_depth;
};

struct RgbdCameraInfo {
    const Camera * rgb;
    const Camera * depth;
};

/// Representation of a frame that the frontend creates for backend processing
struct RgbdFrameEvent: public Event {
    // estimated pose for this frame
    Sophus::SE3d pose;

    // Locations of feature points within the image (relative to left RGB)
    std::vector<cv::KeyPoint> keypoints;

    // Feature descriptions
    cv::Mat descriptions;

    // Frame data
    RgbdFrameData frame_data;

    // Camera info
    RgbdCameraInfo info;
};

// this is defined by the backend
struct KeyframePoseEvent;

// forward declaration
struct ProcessedFrameEvent;

/// Tracking frontend using RGBD camera images
///
/// Color and depth images are not expected to be synchronized. Instead, the `trigger` property
/// can be used to determine if receiving the color or receiving the depth image will trigger
/// processing a frame.
class RgbdFrontend {
public:
    /// Configuration parameters and their defaults
    struct Parameters {
        // number of pixels around existing feature to exclude during feature detection
        int feature_mask_size = 10;

        // parameter for feature detector
        double quality_level = 0.01;

        // spacing requirement for feature detector
        double min_distance = 20.0;

        // max iterations for optical flow calculation
        int flow_iterations_max = 30;

        // error threshold for optical flow calculation
        double flow_threshold = 0.01;

        // the window size around each feature to use for optical flow calculation
        int flow_window_size = 11;

        // the number of mimap levels to generate for optical flow calculation
        int flow_pyramid_levels = 3;

        // ICP iteration limit
        size_t max_iterations = 30;
        
        // ICP sample size
        size_t sample_size = 10;
        
        /// ICP outlier factor
        double outlier_factor = 7.16;

        int num_features = 200;
        int num_features_init = 100;
        int num_features_tracking = 50;
        int num_features_tracking_bad = 20;
        int num_features_needed_for_keyframe = 80;

        // max distance between keyframes
        // Issue: how to determine units?
        double max_keyframe_distance = 10.0;

        // seed value for random number generator
        int seed = 12345;
    };

    /// How lost are we?
    enum class Status {
        /// Just starting up and warming up to first sensor readings
        kInitializing,

        /// Successfully tracking with sufficient frame-to-frame information
        kTracking,

        /// Need to create a new keyframe
        kNewKeyframe,
    };

    /// determine whether color or depth image trigger processing of the next frame
    enum class Trigger: bool {
        /// the color image will trigger processing
        kTriggerColor,

        /// the depth image will trigger processing
        kTriggerDepth
    };

    RgbdFrontend(const Parameters& parameters, const Camera& rgb_camera, const Camera& depth_camera);

    void set_trigger(Trigger trigger) { trigger_ = trigger; }
    Trigger trigger() const { return trigger_; }

    const Parameters& parameters() const { return parameters_; }

    void HandleColorEvent(const ImageEvent& event);
    void HandleDepthEvent(const ImageEvent& event);

    /// Downstream modules subscribe here for notification when a new keyframe is available.
    /// Processing time should be minimal or move to a separate thread.
    EventListenerList<RgbdFrameEvent> keyframes;

    /// Testing and debugging code can subscribe here to get notified when a video frame has been
    /// processed.
    EventListenerList<ProcessedFrameEvent> processed_frames;

    void HandleKeyframePoseEvent(const KeyframePoseEvent& event);

private:
    using KeyPoints = std::vector<cv::KeyPoint>;
    struct KeyframePoseUpdate {
        Timestamp timestamp;
        SE3d previous_pose;
        SE3d new_pose;
    };

    using KeyframePoseUpdates = std::queue<KeyframePoseUpdate>;

    /// Process the next frame and create an updated pose estimation
    void ProcessFrame();

    /// Run the feature detector on the RGB data and extract keyframe feature points
    size_t DetectKeyframeFeatures();

    /// Run the feature detector on the RGB data and extract additional keypoints
    size_t DetectAdditionalFeatures(size_t num_additonal);

    /// Use optical flow to find the key points detected in the previous frame in the current frame
    size_t FindFeaturesInCurrent(std::vector<Point2f>& points);

    /// Track previously identified features in a new frame to process
    size_t TrackFeatures();

    /// Use an estimated camera pose and depth information to predict key point positions
    void PredictFeaturesInCurrent(const SE3d& predicted_pose, std::vector<Point2f>& points);

    // Reset all feature vectors
    void ClearFeatureVectors();

    // Trigger a key frame event using the current frame and tracking information
    void PostKeyframe();

    // Notify subscribed listeners on processing of a new video frame
    void PostProcessedFrame(Timestamp timestamp, Status old_state, Status new_state, const SE3d& pose,
                            size_t num_tracked_features, bool is_keyframe);

    // Various configuration prameters
    Parameters parameters_;

    /// Parameters describing the RGB camera
    const Camera& rgb_camera_;

    /// Parameters describing the depth camera
    const Camera& depth_camera_;

    /// Processing trigger
    Trigger trigger_;

    /// Current processing status
    Status status_;

    /// Timepoint last processed frame
    Timestamp last_processed_time_;

    /// Pose associated with the previous frame
    Sophus::SE3d current_pose_;

    /// Camera data per frame to be processed
    RgbdFrameData current_frame_data_;

    /// Pose associated with the previous frame
    Sophus::SE3d previous_pose_;

    /// Timestamp of last keyframe
    Timestamp last_keyframe_timestamp_;

    /// Pose associated with the most recent key frame
    Sophus::SE3d last_keyframe_pose_;

    /// Distance since last keyframe
    double distance_since_last_keyframe_;

    /// Queue of keyframe pose adjustments coming back from the backend
    KeyframePoseUpdates keyframe_pose_updates_;

    /// Previous frame data
    RgbdFrameData previous_frame_data_;

    /// Keypoints/features tracked
    KeyPoints key_points_;

    /// Keypoints/features tracked
    std::vector<Point2f> tracked_features_;

    /// 3-D coordinates of tracked features
    std::vector<Point3d> tracked_feature_coords_;

    /// Relative motion between previous two frames
    Sophus::SE3d relative_motion_;

    /// Tangent representation (log) of relative_motion_
    Sophus::SE3d::Tangent relative_motion_twist_;
    
    /// Feature detector
    cv::Ptr<cv::FeatureDetector> feature_detector_;

    /// Random number generator to use
    std::default_random_engine random_engine_;
};

/// Representation of a tracking event that the frontend creates upon processing of an
/// incoming frame.
///
/// This is primarily intended for testing and debugging
struct ProcessedFrameEvent: public Event {
    RgbdFrontend::Status old_state, new_state;

    // estimated pose for this frame
    Sophus::SE3d pose;

    // number of tracked features
    size_t num_tracked_features;

    // is this recorded as keyframe?
    bool is_keyframe;
};

} // namespace slammer

#endif //ndef SLAMMER_RGBD_FRONTEND_H
