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

#include "slammer/optimizer.h"

#include "slammer/camera.h"
#include "slammer/pnp.h"
#include "slammer/loris/opencv_utils.h"

#include "boost/gil/extension/io/png.hpp"

using namespace slammer;
using namespace slammer::loris;

namespace {

using namespace boost::gil;

// Depth scale used: https://lifelong-robotic-vision.github.io/dataset/scene
static constexpr double kDepthScale = 0.001;

inline double DepthForPixel(const gray16c_view_t& frame, Point2f point) {
    float column = point.x, row = point.y; 
    auto depth = frame.at(floorf(column), floorf(row))[0];

    // 0 depth represents a missing depth value, so we convert those to an NaN value instead
    return depth ? kDepthScale * depth : std::numeric_limits<double>::signaling_NaN();
}

std::vector<Point3d> ToSpatial(const Camera& camera, const gray16c_view_t& frame,
                               const std::vector<Point2f>& points) {
    const float sigma = 1.0f;

    size_t window = static_cast<size_t>(roundf(sigma * 3)) * 2 + 1;
    auto smoothing = boost::gil::generate_gaussian_kernel(window, sigma);

    gray16_image_t smoothed(frame.dimensions());
    boost::gil::detail::convolve_2d(frame, smoothing, view(smoothed));

    auto smoothed_frame = const_view(smoothed); 
    std::vector<Point3d> result;

    for (size_t index = 0; index < points.size(); ++index) {
        const auto& point = points[index];
        auto z = DepthForPixel(smoothed_frame, point);

        // if (isnan(z)) {
        //     std::cout << "NaN at index " << index << std::endl;
        // }

        result.push_back(camera.PixelToCamera(point, z));
    }

    return result;
}

} // namespace

TEST(OptimizerTest, ReconstructPoseNoError) {
    for (int iter = 0; iter < 2; ++iter) {
        std::default_random_engine random_engine(12345);

        // Determine a camera
        std::string kDataSetPath("data/cafe1-1");
        
        Result<SensorInfo> sensor_info_result = ReadSensorInfo(kDataSetPath);
        EXPECT_TRUE(sensor_info_result.ok());
        auto sensor_info = sensor_info_result.value();

        StereoDepthCamera camera = CreateAlignedStereoDepthCamera(sensor_info.d400_color_optical_frame, 0.05, SE3d());

        // Determine a set of points
        std::vector<Point3d> points = {
            Point3d(4.0, 1.2, 3.5),
            Point3d(1.0, 2.4, 5.5),
            Point3d(6.0, 3.0, 4.5),
            Point3d(3.0, 0.5, 4.2),
            Point3d(5.0, 7.1, 3.6),
            Point3d(4.0, 5.4, 7.2),
        };

        // Determine a second pose
        Quaterniond rotation(Eigen::AngleAxisd(0.3, Point3d(0.0, 0.0, 1.0)));
        Point3d translation(0.2, 0.1, 0.5);
        SE3d pose(rotation, translation);
        SE3d inv_pose = pose.inverse();

        // Project points using camera and pose and camera
        PerspectiveAndPoint3d::PointPairs point_pairs;

        for (const auto& point: points) {
            point_pairs.emplace_back(camera.CameraToPixelDisparity(point), 
                                     camera.CameraToPixelDisparity(inv_pose * point));
        }

        // Reconstruct pose and locations; using true locations and identity pose as starting point
        Eigen::VectorXd variables(points.size() * 3 + 6);

        for (size_t index = 0; index < points.size(); ++index) {
            variables(Eigen::seqN(index * 3 + 6, 3)) = points[index];
        }

        variables(Eigen::seqN(0, 6)) = SE3d::Tangent::Zero();

        PerspectiveAndPoint3d instance(camera, camera, point_pairs);

        using namespace std::placeholders;

        auto result = 
            iter == 0 ?
                GaussNewton(std::bind(&PerspectiveAndPoint3d::CalculateJacobian, &instance, _1),
                            std::bind(&PerspectiveAndPoint3d::CalculateResidual, &instance, _1),
                            variables, 10) :
                LevenbergMarquardt(std::bind(&PerspectiveAndPoint3d::CalculateJacobian, &instance, _1),
                                   std::bind(&PerspectiveAndPoint3d::CalculateResidual, &instance, _1),
                                   variables, 10, 0.01);

        EXPECT_TRUE(result.ok());

        // Reconstructed pose should match preset pose
        SE3d calculated_pose = SE3d::exp(variables(Eigen::seqN(0, 6)));
        const double kEpsilon = 0.00001;

        EXPECT_LE((inv_pose.translation() - calculated_pose.translation()).squaredNorm(), kEpsilon);
        EXPECT_TRUE(calculated_pose.unit_quaternion().isApprox(inv_pose.unit_quaternion(), kEpsilon));

        // Reconstructed point positions should match original positions

        for (size_t index = 0; index < points.size(); ++index) {
            Point3d point = variables(Eigen::seqN(index * 3 + 6, 3));
            EXPECT_LE((point - points[index]).squaredNorm(), kEpsilon);
        }
    }
}


TEST(OptimizerTest, ReconstructPoseWithError) {
    for (int iter = 0; iter < 2; ++iter) {

        std::default_random_engine random_engine(12345);

        // Determine a camera
        std::string kDataSetPath("data/cafe1-1");
        
        Result<SensorInfo> sensor_info_result = ReadSensorInfo(kDataSetPath);
        EXPECT_TRUE(sensor_info_result.ok());
        auto sensor_info = sensor_info_result.value();

        StereoDepthCamera camera = CreateAlignedStereoDepthCamera(sensor_info.d400_color_optical_frame, 0.05, SE3d());

        // Determine a set of points
        std::vector<Point3d> points = {
            Point3d(4.0, 1.2, 3.5),
            Point3d(1.0, 2.4, 5.5),
            Point3d(6.0, 3.0, 4.5),
            Point3d(3.0, 0.5, 4.2),
            Point3d(5.0, 7.1, 3.6),
            Point3d(4.0, 5.4, 7.2),
        };

        // Determine a second pose
        Quaterniond rotation(Eigen::AngleAxisd(0.5, Point3d(0.0, 0.0, 1.0)));
        Point3d translation(1.2, 1.0, 0.5);
        SE3d pose(rotation, translation);
        SE3d inv_pose = pose.inverse();

        // Project points using camera and pose and camera
        PerspectiveAndPoint3d::PointPairs point_pairs;
        std::normal_distribution<double> normal(0.0, 0.1);

        for (const auto& point: points) {
            Vector3d first_noise(normal(random_engine), normal(random_engine), normal(random_engine));
            Vector3d second_noise(normal(random_engine), normal(random_engine), normal(random_engine));

            point_pairs.emplace_back(camera.CameraToPixelDisparity(point) + first_noise, 
                                     camera.CameraToPixelDisparity(inv_pose * point) + second_noise);
        }

        // Reconstruct pose and locations; using true locations and identity pose as starting point
        Eigen::VectorXd variables(points.size() * 3 + 6);

        for (size_t index = 0; index < points.size(); ++index) {
            variables(Eigen::seqN(index * 3 + 6, 3)) = points[index];
        }

        variables(Eigen::seqN(0, 6)) = SE3d::Tangent::Zero();

        PerspectiveAndPoint3d instance(camera, camera, point_pairs);

        using namespace std::placeholders;

        auto result = 
            iter == 0 ?
                GaussNewton(std::bind(&PerspectiveAndPoint3d::CalculateJacobian, &instance, _1),
                            std::bind(&PerspectiveAndPoint3d::CalculateResidual, &instance, _1),
                            variables, 10) :
                LevenbergMarquardt(std::bind(&PerspectiveAndPoint3d::CalculateJacobian, &instance, _1),
                                   std::bind(&PerspectiveAndPoint3d::CalculateResidual, &instance, _1),
                                   variables, 10, 0.01);

        EXPECT_TRUE(result.ok());

        // Reconstructed pose should match preset pose
        SE3d calculated_pose = SE3d::exp(variables(Eigen::seqN(0, 6)));
        const double kEpsilon = 0.05;

        EXPECT_LE((inv_pose.translation() - calculated_pose.translation()).squaredNorm(), kEpsilon);
        EXPECT_TRUE(calculated_pose.unit_quaternion().isApprox(inv_pose.unit_quaternion(), kEpsilon));

        // Reconstructed point positions should match original positions

        for (size_t index = 0; index < points.size(); ++index) {
            Point3d point = variables(Eigen::seqN(index * 3 + 6, 3));
            EXPECT_LE((point - points[index]).squaredNorm(), kEpsilon);
        }
    }
}

TEST(OptimizerTest, PoseFromFeatures) {
    using namespace boost::gil;

    // ICP iteration limit
    size_t max_iterations = 30;
    
    // ICP sample size
    size_t sample_size = 10;
    
    /// ICP outlier threshold
    double outlier_threshold = 1.0;

    std::default_random_engine random_engine(12345);

    std::string kDataSetPath("data/cafe1-1");
    
    Result<SensorInfo> sensor_info_result = ReadSensorInfo(kDataSetPath);
    EXPECT_TRUE(sensor_info_result.ok());
    auto sensor_info = sensor_info_result.value();

    StereoDepthCamera depth_camera = CreateAlignedStereoDepthCamera(sensor_info.d400_depth_optical_frame, 0.05, SE3d());

    std::string kSourceInputPath("data/cafe1-1/aligned_depth/1560004885.446165.png");
    //std::string kTargetInputPath("data/cafe1-1/aligned_depth/1560004886.446849.png");
    std::string kTargetInputPath("data/cafe1-1/aligned_depth/1560004887.013907.png");

    // #Time px py pz qx qy qz qw
    // 1560004885.44099879 21.37940216064453 21.128952026367188 0.0 0.0 0.0 -0.30020832668341807 0.9538736607066692
    // 1560004886.44052076 21.386783599853516 21.11459732055664 0.0 0.0 0.0 -0.3022010165694693 0.9532442213747636
    // 1560004887.01535177 21.385334014892578 21.11564826965332 0.0 0.0 0.0 -0.30227737870033583 0.9532200094028944

    gray16_image_t source_input, target_input;
    read_image(kSourceInputPath, source_input, png_tag{});
    read_image(kTargetInputPath, target_input, png_tag{});

    std::vector<Point2f> source_points = {
        Point2f(372, 232),         Point2f(368, 240),
        Point2f(352, 280),         Point2f(311.127, 234.759),
        Point2f(420, 228),         Point2f(285.671, 322.441),
        Point2f(336.583, 285.671), Point2f(620, 208),
        Point2f(300, 236),         Point2f(336.583, 316.784),
        Point2f(356, 292),         Point2f(404, 148),
        Point2f(610, 234),         Point2f(652, 104),
        Point2f(418.607, 237.588), Point2f(352, 272),
        Point2f(316, 244),         Point2f(370, 300),
        Point2f(292, 276),         Point2f(358, 300),
        Point2f(328, 168),         Point2f(684, 112),
        Point2f(352, 336),         Point2f(720, 344),
        Point2f(622.254, 79.196),  Point2f(345.068, 254.558),
        Point2f(707.107, 330.926), Point2f(380, 132),
        Point2f(356, 400),         Point2f(296, 316),
        Point2f(333.754, 200.818), Point2f(328.098, 175.362),
        Point2f(708, 156),         Point2f(325.269, 285.671),
        Point2f(256, 248),         Point2f(695.793, 104.652),
        Point2f(792, 248),         Point2f(299.813, 299.813),
        Point2f(364, 250),         Point2f(408, 268),
        Point2f(200, 232),         Point2f(257.387, 313.955),
        Point2f(344, 244),         Point2f(676, 284),
        Point2f(421.436, 253.144), Point2f(296, 256),
        Point2f(348, 224)
    };

    std::vector<Point2f> target_points = {
        Point2f(365.061, 230.996), Point2f(361.496, 238.838),
        Point2f(348.167, 279.567), Point2f(307.692, 233.694),
        Point2f(416.851, 226.734), Point2f(280.675, 321.57),
        Point2f(333.135, 284.835), Point2f(622.131, 204.714),
        Point2f(296.393, 233.275), Point2f(332.399, 315.882),
        Point2f(352.65, 290.952),  Point2f(400.582, 146.884),
        Point2f(601.865, 224.568), Point2f(653.951, 101.983),
        Point2f(415.215, 237.144), Point2f(348.062, 271.404),
        Point2f(312.694, 242.996), Point2f(366.36, 299.459),
        Point2f(288.131, 275.088), Point2f(354.403, 298.186),
        Point2f(324.751, 166.939), Point2f(681.287, 110.982),
        Point2f(347.431, 335.237), Point2f(717.785, 343.487),
        Point2f(630.973, 76.1746), Point2f(341.592, 253.424),
        Point2f(704.922, 329.642), Point2f(376.707, 130.859),
        Point2f(351.726, 399.863), Point2f(291.599, 315.185),
        Point2f(330.337, 199.391), Point2f(324.395, 174.067),
        Point2f(704.974, 154.918), Point2f(321.719, 284.578),
        Point2f(251.991, 247.178), Point2f(692.747, 103.471),
        Point2f(788.897, 247.241), Point2f(296.252, 299.118),
        Point2f(358.315, 250.245), Point2f(404.556, 266.896),
        Point2f(195.985, 230.787), Point2f(252.838, 313.211),
        Point2f(340.375, 242.485), Point2f(673.121, 282.322),
        Point2f(418.119, 252.123), Point2f(292.101, 255.078),
        Point2f(344.338, 223.113)
    };

    ASSERT_EQ(source_points.size(), target_points.size());

    auto source_spatial = ToSpatial(depth_camera, const_view(source_input), source_points);
    auto target_spatial = ToSpatial(depth_camera, const_view(target_input), target_points);

    // {
    //     std::vector<double> differences;
    //     double sum = 0;
    //     for (size_t index = 0; index < source_spatial.size(); ++index) {
    //         double diff = target_spatial[index].z() - source_spatial[index].z();
    //         sum += diff;
    //         differences.push_back(diff);
    //     }

    //     std::cout << "Average delta Z = " << (sum/source_spatial.size()) << std::endl;
    //     std::sort(differences.begin(), differences.end());
    //     std::cout << "Perc 25 delta Z = " << (differences[source_spatial.size() * 0.25]) << std::endl;
    //     std::cout << "Perc 40 delta Z = " << (differences[source_spatial.size() * 0.40]) << std::endl;
    //     std::cout << "Perc 50 delta Z = " << (differences[source_spatial.size() * 0.50]) << std::endl;
    //     std::cout << "Perc 60 delta Z = " << (differences[source_spatial.size() * 0.60]) << std::endl;
    //     std::cout << "Perc 75 delta Z = " << (differences[source_spatial.size() * 0.75]) << std::endl;
    // }

    std::vector<uchar> mask(source_spatial.size(), 255);

    // Project points using camera and pose and camera
    PerspectiveAndPoint3d::PointPairs point_pairs;

    for (size_t index = 0; index < source_spatial.size(); ++index) {
        point_pairs.emplace_back(depth_camera.CameraToPixelDisparity(source_spatial[index]), 
                                 depth_camera.CameraToPixelDisparity(target_spatial[index]));
    }

    PerspectiveAndPoint3d instance(depth_camera, depth_camera, point_pairs);
    std::vector<uchar> inliers;

    SE3d calculated_pose;
    const double kThreshold = 7.18;
    auto result = instance.Ransac(calculated_pose, inliers, source_spatial, 10, 20, 0.01, kThreshold, random_engine);
    EXPECT_TRUE(result.ok());

    // Reconstructed pose should match preset pose
    const double kEpsilon = 0.00001;

    SE3d relative_motion = calculated_pose.inverse();

    // Now compare the calculated distance and compare to the baseline; 0.0145663
    Vector3d source_pos(21.37940216064453, 21.128952026367188, 0.0);
    Quaterniond source_rot(0.9538736607066692, 0.0, 0.0, -0.30020832668341807);
    Vector3d target_pos(21.385334014892578, 21.11564826965332, 0.0);
    Quaterniond target_rot(0.9532200094028944, 0.0, 0.0, -0.30227737870033583);

    // 0.0145663
    auto true_distance = (target_pos - source_pos).norm();

    // 0.0158906
    auto estimated_distance = relative_motion.translation().norm();

    // std::cout << "Estimated rotation: " << relative_motion.unit_quaternion().coeffs() << std::endl;
    // std::cout << "True rotation: " << (target_rot * source_rot.inverse()).coeffs() << std::endl;

    // estimate vs atual is less than 1 cm
    EXPECT_LE(fabs(estimated_distance - true_distance), 0.01);

    for (size_t index = 0; index < source_points.size(); ++index) {
        const auto& point_pair = point_pairs[index];
        auto prediction = 
            depth_camera.CameraToPixelDisparity(calculated_pose *
                                                depth_camera.PixelDisparityToCamera(point_pair.first));

        double error = (point_pair.second - prediction).squaredNorm();

        // inliers/outlier deparation within 20% around threshold
        auto thresholdLow = kThreshold * 0.8, thresholdHigh = kThreshold * 1.2;

        if (!inliers[index]) {
            EXPECT_GT(error, thresholdLow);

            if (error <= thresholdLow) {
                std::cout << "Invalid outlier at index " << index << std::endl;
            }
        } else {
            EXPECT_LE(error, thresholdHigh);

            if (error > thresholdHigh) {
                std::cout << "Invalid inlier at index " << index << std::endl;
            }
        }
    }
}