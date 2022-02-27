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
#include "slammer/loris/opencv_utils.h"

using namespace slammer;
using namespace slammer::loris;

namespace {

template <typename Matrix, typename Iterator>
void EmitTriplets(const Matrix& matrix, Iterator output, size_t row_offset, size_t column_offset) {
    for (size_t row_index = 0; row_index < matrix.rows(); ++row_index) {
        for (size_t column_index = 0; column_index < matrix.cols(); ++column_index) {
            auto value = matrix(row_index, column_index);

            if (value) {
                *output++ = Eigen::Triplet<double>(row_offset + row_index, column_offset + column_index, value);
            }
        }
    }
}

}

/// Instance of a perspective and point problem for two sets of point generated with
/// an RGB-D camera.
///
/// We are given two sets of coordinates, where entries at the same index correspond
/// to measurements of the same feature in 3D from two camera locations. The (x,y)
/// coordinates are the 2D image location within a standard (RGB) camera image, and
/// the z coordinate is the depth sensor value. For the model here, we assume that 
/// the depth value has been derived by determining the dispersion across a pair of
/// stereo images, as is, for example, the case for an Intel Realsense camera.
/// We also assume that the first camera is placed at the origin and looking in 
/// (negative?) Z direction.
class PerspectiveAndPoint3d {
public:
    typedef std::pair<Point3d, Point3d> PointPair;
    typedef std::vector<PointPair> PointPairs;

    PerspectiveAndPoint3d(const StereoDepthCamera& first, const StereoDepthCamera& second,
                          const PointPairs& point_pairs)
        : first_(first), second_(second), point_pairs_(point_pairs) {

    }

    /// Calculate the Jacobian of the problem instance
    /// The rows of the Jacobian correspond to the measurement values associated with the 
    /// point pairs.
    ///
    /// That is, the rows are blocks of six rows each, in the order of the provided point
    /// pairs. Each block has two parts corresponding to the 3 coordinates of the first
    /// measurement followed by the 3 coordinates of the second measurement.
    ///
    /// The columns correspond to the vector variables that we are optimizing. The initial
    /// 6 define the tangent corresponding via exponential map to the pose in SE(3)
    /// associated with the second camera. Per documentation of ``Sophus::SE3::exp()``,
    /// the first three components of those coordinates represent the translational part
    /// ``upsilon`` in the tangent space of SE(3), while the last three components
    /// of ``a`` represents the rotation vector ``omega``.
    ///
    /// Then, for each point pair, we have an entry corresponding to (x, y, z) coordinates 
    /// of the meassured feature point in 3D world space.
    Eigen::SparseMatrix<double> CalculateJacobian(const Eigen::VectorXd& value) const {
        SE3d::Tangent tangent(value(Eigen::seqN(0, 6)));
        SE3d second_pose = SE3d::exp(tangent);

        Eigen::Matrix<double, 3, 3> jacobian_second_coords = second_pose.so3().matrix();

        const size_t rows = point_pairs_.size() * 6;
        const size_t columns = point_pairs_.size() * 3 + 6;
        using Triplet = Eigen::Triplet<double>;
        std::vector<Triplet> triplets;

        // fill in the coefficients of the matrix
        for (size_t index = 0; index < point_pairs_.size(); ++index) {
            Vector3d coords(value(Eigen::seqN(index * 3 + 6, 3)));

            // generate entries for first measurement error depending on first measurement
            // this is just the Jacobian of mapping the point coordinates via the camera
            // to pixel coordinates.
            auto jacobian = first_.Jacobian(coords);

            EmitTriplets(jacobian, std::back_insert_iterator(triplets), index * 6, index * 3 + 6);

            // generate entries for second measurement error depending on second measurement
            Point3d transformed = second_pose * coords;
            Matrix3d jacobian_project = second_.Jacobian(transformed);
            Matrix3d jacobian_coords = jacobian_project * second_pose.so3().matrix();

            EmitTriplets(jacobian_coords, std::back_insert_iterator(triplets), index * 6 + 3, index * 3 + 6);

            // generate entries for second measurement error depending on second pose
            Eigen::Matrix<double, 3, 6> jacobian_second_params;
            jacobian_second_params(Eigen::seqN(0, 3), Eigen::seqN(0, 3)) = Matrix3d::Identity();
            jacobian_second_params(Eigen::seqN(0, 3), Eigen::seqN(3, 3)) = -SE3d::SO3Member::hat(transformed);
            Eigen::Matrix<double, 3, 6> jacobian_params = jacobian_project * jacobian_second_params;

            EmitTriplets(jacobian_params, std::back_insert_iterator(triplets), index * 6 + 3, 0);
        }

        Eigen::SparseMatrix<double> result(rows, columns);
        result.setFromTriplets(triplets.begin(), triplets.end());
        return result;
    }

    Eigen::VectorXd CalculateResidual(const Eigen::VectorXd& value) const {
        SE3d::Tangent tangent(value(Eigen::seqN(0, 6)));
        SE3d second_pose = SE3d::exp(tangent);

        Eigen::VectorXd residual(point_pairs_.size() * 6);

        for (size_t index = 0; index < point_pairs_.size(); ++index) {
            Vector3d coords(value(Eigen::seqN(index * 3 + 6, 3)));
            auto first_projection = first_.CameraToPixelDisparity(coords);
            residual(Eigen::seqN(index * 6, 3)) = first_projection - point_pairs_[index].first;

            auto second_coords = second_pose * coords;
            auto second_projection = second_.CameraToPixelDisparity(second_coords);
            residual(Eigen::seqN(index * 6 + 3, 3)) = second_projection - point_pairs_[index].second;
        }

        return residual;
    }

private:
    StereoDepthCamera first_;
    StereoDepthCamera second_;
    PointPairs point_pairs_;
};

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
                                   variables, 10, 0.1);

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
                                   variables, 10, 0.1);

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


