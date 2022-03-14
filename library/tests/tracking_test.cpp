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

#include "slammer/slammer.h"
#include "slammer/flow.h"
#include "slammer/orb.h"
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

class TrackingListener {
public:
    TrackingListener(const std::string& path)
      : image_logger_(path) {}

    void HandleGroundtruthEvent(const loris::GroundtruthEvent& event) {
        current_groundtruth_pose = SE3d(event.orientation, Point3d::Zero()) * 
            SE3d(SE3d::QuaternionType::Identity(), event.position);;
    }

    void HandleInitializedEvent(const InitializedEvent& event) {
        ++num_frames;
        current_timestamp = event.timestamp;
        refernce_timestamp = event.timestamp;
        reference_pose = event.pose;
        reference_groundtruth_pose = current_groundtruth_pose;

        auto gray = RgbToGrayscale(const_view(*event.image));
        LogFeatures(image_logger_, fmt::format("initialized-{}.", event.timestamp.time_since_epoch().count()), 
                    const_view(gray), event.features);
    }

    void HandleTrackingEvent(const TrackingEvent& event) {
        ++num_frames;
        is_tracking = true;
        current_timestamp = event.timestamp;
        current_pose = event.pose;

        auto groundtruth_translation = (current_groundtruth_pose * reference_groundtruth_pose.inverse()).translation();
        auto groundtruth_distance = groundtruth_translation.norm();
        auto estimated_translation = (current_pose * reference_pose.inverse()).translation();
        auto estimated_distance = estimated_translation.norm();

        // should be within 90%?
        auto diff = estimated_distance - groundtruth_distance;
        EXPECT_LE(fabs(diff), 0.1);

        auto factor = estimated_distance / groundtruth_distance;

        EXPECT_GE(factor, 0.75);
        EXPECT_LE(factor, 1.25);

        // EXPECT_EQ(estimated_distance, groundtruth_distance);

        auto gray = RgbToGrayscale(const_view(*event.image));
        LogFeatures(image_logger_, fmt::format("tracking-{}.", event.timestamp.time_since_epoch().count()), 
                    const_view(gray), event.features, &event.reference);
    }

    void HandleTrackingLostEvent(const TrackingLostEvent& event) {
        current_timestamp = event.timestamp;
        is_tracking = false;
    }

    FileImageLogger image_logger_;

    size_t num_frames = 0;
    bool is_tracking = false;

    SE3d current_groundtruth_pose;
    SE3d reference_groundtruth_pose;
    SE3d reference_pose;
    SE3d current_pose;
    Timestamp refernce_timestamp;
    Timestamp current_timestamp;
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

    /// We have been unable to track features any further
    kTrackingLost
};

struct AbsoluteTrackerParameters {
    // which image event does trigger frame processing
    Trigger trigger = Trigger::kTriggerColor;

    // number of frames to skip between processing
    unsigned skip_count = 0;

    // seed value for random number generator
    int seed = 12345;

    // parameters for the ORB detector
    orb::Parameters orb_parameters;

    // how many features would we like to detect and track overall 
    size_t num_features = 200;

    size_t num_features_init = 100;
    size_t num_features_tracking = 50;
    size_t num_features_tracking_bad = 20;
    size_t num_features_needed_for_keyframe = 80;

    // max iterations for optical flow calculation
    int flow_iterations_max = 30;

    // error threshold for optical flow calculation
    double flow_threshold = 0.5;

    // the half-window size around each feature to use for optical flow calculation
    int flow_omega = 3;

    // the number of mimap levels to generate for optical flow calculation
    int flow_pyramid_levels = 4;

    // RANSAC iteration limit
    size_t max_iterations = 30;
    
    // RANSAC sample size
    size_t sample_size = 6;
    
    /// RANSAC outlier threshold
    double outlier_threshold = 7.81;

    /// Levenberg-Marquardt initial lambda parameter for PnP calculation  
    double lambda = 0.01;
};

class AbsoluteTracker {
public:
    AbsoluteTracker(const AbsoluteTrackerParameters& parameters, 
                    const Camera& rgb_camera, const StereoDepthCamera& depth_camera)
        : parameters_(parameters), rgb_camera_(rgb_camera), depth_camera_(depth_camera),
          state_(TrackingState::kInitializing), detector_(parameters.orb_parameters),
          skip_count_(0), random_engine_(parameters.seed) {}

    /// Event handler for an incoming color image from the RGB camera
    ///
    /// \param event the event data structure procviding access to the image data
    void HandleColorEvent(const ColorImageEvent& event) {
        current_state_.color_image = event.image;

        if (parameters_.trigger == Trigger::kTriggerColor) {
            ProcessCurrentImages(event.timestamp);
        }
    }

    /// Event handler for an incoming depth image from the depth camera
    ///
    /// \param event the event data structure procviding access to the image data
    void HandleDepthEvent(const DepthImageEvent& event) {
        current_state_.depth_image = event.image;

        if (parameters_.trigger == Trigger::kTriggerDepth) {
            ProcessCurrentImages(event.timestamp);
        }
    }

    EventListenerList<InitializedEvent> initialization;
    EventListenerList<TrackingEvent> tracking;
    EventListenerList<TrackingLostEvent> tracking_lost;

protected:
    void ProcessCurrentImages(Timestamp timestamp) {
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
            DoTracking(timestamp);
            break;

        case TrackingState::kTrackingLost:
            DoRecover(timestamp);
            break;
        }
    }

    void DidInitialize() {
        InitializedEvent event {
            reference_state_.timestamp,
            reference_state_.pose,
            reference_state_.pixel_coords.size(),
            reference_state_.color_image,
            reference_state_.pixel_coords
        };

        initialization.HandleEvent(event);
    }

    void DoInitialize(Timestamp timestamp) {
        std::vector<orb::KeyPoint> key_points;

        detector_.ComputeFeatures(const_view(*current_state_.color_image), 
                                  parameters_.num_features, key_points);

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

        // keep track of time
        reference_state_.timestamp = current_state_.timestamp = timestamp;

        // initialize the current and reference pose to origin; this may need to changed later
        current_state_.pose = reference_state_.pose = SE3d();

        // initialize the "velocity" to zero
        relative_motion_twist_.setZero();

        // update state 
        state_ = TrackingState::kTracking;

        DidInitialize();
    }

    void DidTrack() {
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

    void LostTracking() {
        TrackingLostEvent event {
            current_state_.timestamp,
        };

        tracking_lost.HandleEvent(event);
    }

    void DoTracking(Timestamp timestamp) {
        std::vector<float> error;

        // TODO: Refactor this such that the same conversion can be used for feature detection
        auto source = RgbToGrayscale(const_view(*reference_state_.color_image));
        auto target = RgbToGrayscale(const_view(*current_state_.color_image));

        auto estimated_motion =
            SE3d::exp(relative_motion_twist_ * (timestamp - current_state_.timestamp).count());
        auto estimated_transform = (estimated_motion * current_state_.pose).inverse();

        current_state_.pixel_coords.clear();
        std::transform(reference_state_.camera_coords.begin(), reference_state_.camera_coords.end(),
            std::back_insert_iterator(current_state_.pixel_coords),
            [&](const auto& point) { return rgb_camera_.CameraToPixel(estimated_transform * point); });

        // TODO: initialize the target coordinates using the motion estimate captured in
        // relative_motion_twist_
        // current_state_.pixel_coords = reference_state_.pixel_coords;

        ComputeFlow(const_view(source), const_view(target),
                    reference_state_.pixel_coords, current_state_.pixel_coords, error,
                    parameters_.flow_pyramid_levels, parameters_.flow_omega, parameters_.flow_threshold);

        float squared_error = parameters_.flow_threshold * parameters_.flow_threshold;

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
        SE3d calculated_pose = reference_state_.pose.inverse();

        auto result =
            instance.Ransac(calculated_pose, mask, reference_state_.camera_coords, 
                            parameters_.sample_size, parameters_.max_iterations, parameters_.lambda, 
                            parameters_.outlier_threshold, random_engine_);

        if (!result.ok()) {
            state_ = TrackingState::kTrackingLost;
            LostTracking();
            return;
        }

        auto new_pose = calculated_pose.inverse() * reference_state_.pose;
        relative_motion_twist_ = 
            -(current_state_.pose * new_pose.inverse()).log() * (1.0/(timestamp - current_state_.timestamp).count());
        current_state_.pose = new_pose;
        current_state_.timestamp = timestamp;

        Compress(current_state_.pixel_coords, mask);
        Compress(reference_state_.pixel_coords, mask);
        Compress(reference_state_.camera_coords, mask);

        if (reference_state_.pixel_coords.size() < parameters_.num_features_tracking_bad) {
            state_ = TrackingState::kTrackingLost;
            LostTracking();
            return;
        }

        DidTrack();
    }

    void DoRecover(Timestamp timestamp) {

    }

private:
    AbsoluteTrackerParameters parameters_;
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
TEST(TrackingTest, TestAbsolute) {
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

    AbsoluteTrackerParameters parameters;
    AbsoluteTracker tracker(parameters, rgb_camera, depth_camera);
    driver.color.AddHandler(std::bind(&AbsoluteTracker::HandleColorEvent, &tracker, _1));
    driver.aligned_depth.AddHandler(std::bind(&AbsoluteTracker::HandleDepthEvent, &tracker, _1));

    TrackingListener listener("image_logs/tracking_test/test_absolute");
    driver.groundtruth.AddHandler(std::bind(&TrackingListener::HandleGroundtruthEvent, &listener, _1));
    tracker.initialization.AddHandler(std::bind(&TrackingListener::HandleInitializedEvent, &listener, _1));
    tracker.tracking.AddHandler(std::bind(&TrackingListener::HandleTrackingEvent, &listener, _1));
    tracker.tracking_lost.AddHandler(std::bind(&TrackingListener::HandleTrackingLostEvent, &listener, _1));

    // run for 5 secs of simulated events
    auto result = driver.Run(slammer::Timediff(5.0));
    EXPECT_TRUE(result.ok());
    EXPECT_TRUE(listener.is_tracking);
    EXPECT_EQ(listener.num_frames, 149);
}

namespace {

} // namespace

// Test tracking by determining incremental pose changes between adjacent frames
//
// Debugging information to generate along the way:
//  - Image logs showing how features are mapped
//  - Pose relative to first frame as calculated
//  - Pose relative to first frame as provided by ground truth
TEST(TrackingTest, TestRelative) {
    // Determine features and 3d coordinates in first frame
    // Store this information as previous state


    // for each subsequent frame: 
    //  Use optical flow to determine updated locations of feature points
    //  Solve PnP problem to estimate a pose relative to the previous frame

}