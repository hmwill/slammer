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
#include "slammer/orb.h"

namespace slammer {

struct TrackingLostEvent: public Event {

};

struct TrackingEvent: public Event {
    SE3d pose;
    size_t num_features;
    ColorImage image;
    std::vector<Point2f> features;
    std::vector<Point2f> reference;
};

struct InitializedEvent: public Event {
    SE3d pose;
    size_t num_features;
    ColorImage image;
    std::vector<Point2f> features;
};

/// Representation of a keyframe that the tracking frontend creates for backend processing
struct KeyframeEvent: public Event {
    // estimated pose for this frame
    Sophus::SE3d pose;

    // Locations of feature points within the image (relative to left RGB)
    std::vector<Point2f> keypoints;

    // Estimated 3d positions of features in camera coordinates
    std::vector<Point3d> keypoint_coords;

    // Descriptors associated with the feature points
    Descriptors descriptions;

    // Color frame data
    ColorImage color;

    // Depth frame data
    DepthImage depth;

    // Color camera info
    const Camera& rgb_camera;

    // Depth camera info
    const StereoDepthCamera& depth_camera;
};

struct TrackerState {
    // the time represented in this state
    Timestamp timestamp;

    // the pose associated with this state
    SE3d pose;

    // the color image associated with this state
    ColorImage color_image;

    // the depth image associated with this state
    DepthImage depth_image;

    // features as tracked in the color/depth image
    std::vector<Point2f> pixel_coords;

    // features as tracked in 3d camera space
    std::vector<Point3d> camera_coords; 
};

/// determine whether color or depth image trigger processing of the next frame
enum class Trigger: bool {
    /// the color image will trigger processing
    kTriggerColor,

    /// the depth image will trigger processing
    kTriggerDepth
};

/// The state of the tracker
enum TrackingState {
    /// We are not tracking yet
    kInitializing,

    /// We have features that we attempt to track frame to frame
    kTracking,

    /// We are still tracking, but would like to acquire a new reference frame
    kTrackingBad,

    /// We have been unable to track features any further
    kTrackingLost
};

/// Parameters to tune optical flow calculation for feature tracking
struct FlowParameters {
    // max iterations for optical flow calculation
    int max_iterations = 10;

    // error threshold for optical flow calculation
    double threshold = 0.5;

    // the half-window size around each feature to use for optical flow calculation
    int omega = 3;

    // the number of mimap levels to generate for optical flow calculation
    int pyramid_levels = 4;
};

/// Parameters for perspective and points calculation
struct PnpParameters {
    // RANSAC iteration limit
    size_t max_iterations = 30;
    
    // RANSAC sample size
    size_t sample_size = 6;
    
    /// RANSAC outlier threshold
    double outlier_threshold = 7.81;

    /// Levenberg-Marquardt initial lambda parameter for PnP calculation  
    double lambda = 0.01;
};

/// Parameters governing tracking behavior
struct TrackingParameters {
    /// how many features would we like to detect and track overall 
    size_t num_features = 200;

    size_t num_features_init = 100;
    size_t num_features_tracking = 50;
    size_t num_features_tracking_bad = 20;
    size_t num_features_needed_for_keyframe = 80;

    /// maximum tracking duration (in seconds)
    Timediff max_duration = Timediff(2.0);

    /// maximum tracking distance (in meters)
    double max_distance = 2.0;
};

struct FrontendParameters {
    // which image event does trigger frame processing
    Trigger trigger = Trigger::kTriggerColor;

    // number of frames to skip between processing
    unsigned skip_count = 0;

    // seed value for random number generator
    int seed = 12345;

    // parameters for the ORB detector
    orb::Parameters orb_parameters;

    // parameters for optical flow
    FlowParameters flow;

    // Parameters for perspective and point optimization
    PnpParameters pnp;

    // Parameters governing tracking behavior
    TrackingParameters tracking;
};

class AbsoluteTracker {
public:
    AbsoluteTracker(const FrontendParameters& parameters, 
                    const Camera& rgb_camera, const StereoDepthCamera& depth_camera)
        : parameters_(parameters), rgb_camera_(rgb_camera), depth_camera_(depth_camera),
          state_(TrackingState::kInitializing), detector_(parameters.orb_parameters),
          skip_count_(0), random_engine_(parameters.seed) {}

    /// Event handler for an incoming color image from the RGB camera
    ///
    /// \param event the event data structure procviding access to the image data
    void HandleColorEvent(const ColorImageEvent& event);

    /// Event handler for an incoming depth image from the depth camera
    ///
    /// \param event the event data structure procviding access to the image data
    void HandleDepthEvent(const DepthImageEvent& event);

    EventListenerList<InitializedEvent> initialization;
    EventListenerList<TrackingEvent> tracking;
    EventListenerList<TrackingLostEvent> tracking_lost;
    EventListenerList<KeyframeEvent> keyframes;

protected:
    void ProcessCurrentImages(Timestamp timestamp);
    void DidInitialize();
    bool ReferenceFromCurrent();

    void DoInitialize(Timestamp timestamp);

    void DidTrack();

    void LostTracking();

    void DoTracking(Timestamp timestamp);

    void DoRecover(Timestamp timestamp);

    // Post a new keyframe
    void PostKeyframe(const TrackerState& state, const Descriptors& descriptors);

    // Create a keyframe event from a tracking state
    KeyframeEvent CreateKeyFrameEvent(const TrackerState& state, const Descriptors& descriptors);

private:
    FrontendParameters parameters_;
    TrackingState state_;
    unsigned skip_count_;

    TrackerState reference_state_;
    TrackerState current_state_;

    // current motion "velocity"
    Sophus::SE3d::Tangent relative_motion_twist_;

    Camera rgb_camera_;
    StereoDepthCamera depth_camera_;

    orb::Detector detector_;

    /// Random number generator to use
    std::default_random_engine random_engine_;
};

} // namespace slammer

#endif //ndef SLAMMER_RGBD_FRONTEND_H
