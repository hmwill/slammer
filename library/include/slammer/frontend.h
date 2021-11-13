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

/// Representation of a frame that the frontend creates for backend processing
struct RgbdFrameEvent: public Event {
    // estimated pose for this frame
    Sophus::SE3d pose;

    // Locations of feature points within the image (relative to left RGB)
    std::vector<Point2f> features;

    // Frame data
    RgbdFrameData frame_data;
};

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

        int num_features = 200;
        int num_features_init = 100;
        int num_features_tracking = 50;
        int num_features_tracking_bad = 20;
        int num_features_needed_for_keyframe = 80;
    };

    /// How lost are we?
    enum class Status {
        /// Just starting up and warming up to first sensor readings
        kInitializing,

        /// Successfully tracking with sufficient frame-to-frame information
        kTracking,

        /// Insufficient information carry-over across frames; need to start up again
        kTrackingLost,
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

    void HandleColorEvent(const ImageEvent& event);
    void HandleDepthEvent(const ImageEvent& event);

    /// Downstream modules subscribe here for notification when a new keyframe is available.
    /// Processing time should be minimal or move to a separate thread.
    EventListenerList<RgbdFrameEvent> keyframes;

private:
    using KeyPoints = std::vector<cv::KeyPoint>;

    /// Process the next frame and create an updated pose estimation
    void ProcessFrame();

    /// Run the feature detector on the RGB data and extract the list of keypoints
    void DetectFeatures(const RgbdFrameData& frame_data, KeyPoints& key_points);

    /// Use optical flow to find the key points detected in the previous frame in the current frame
    void FindFeatureInCurrent(std::vector<Point2f>& points, std::vector<unsigned char>& mask);

    Parameters parameters_;

    /// Parameters describing the RGB camera
    Camera rgb_camera_;

    /// Parameters describing the depth camera
    Camera depth_camera_;

    /// Processing trigger
    Trigger trigger_;

    /// Current processing status
    Status status_;

    /// Camera data per frame to be processed
    RgbdFrameData current_frame_data_;

    /// Previous frame data
    RgbdFrameData previous_frame_data_;

    /// Keypoints/features tracked in previous frame
    KeyPoints previous_key_points_;

    /// Feature detector
    ///
    /// This is the OpenCV implementation of
    /// Shi, Jianbo, and Tomasi. 1994. “Good Features to Track.” In 1994 Proceedings of IEEE Conference on Computer 
    /// Vision and Pattern Recognition, 593–600.
    cv::Ptr<cv::GFTTDetector> feature_detector_;
};

} // namespace slammer

#endif //ndef SLAMMER_RGBD_FRONTEND_H
