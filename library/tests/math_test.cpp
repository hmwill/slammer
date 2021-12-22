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

#include "slammer/math.h"

#include "Eigen/Geometry"

using namespace slammer;

TEST(MathTest, TestIcp) {
    std::default_random_engine random_engine(1234);
    std::uniform_real_distribution<double> uniform_distribution(-10.0, 10.0);
    std::uniform_real_distribution<double> noise_distribution(-0.1, 0.1);

    for (double noise_level = 0.0; noise_level <= 1.0; noise_level += 1.0) {
        std::vector<Point3d> reference, transformed;
        Eigen::AngleAxisd angle_axis(0.34, Eigen::Vector3d(0.5, 0.3, 0.1).normalized());
        SE3d::SO3Type::Transformation rotation = angle_axis.toRotationMatrix();
        SE3d::TranslationType translation { 1.0, 2.0, 3.0 };
        SE3d transformation(rotation, translation);

        for (size_t index = 0; index < 10; ++index) {
            double x = uniform_distribution(random_engine);
            double y = uniform_distribution(random_engine);
            double z = uniform_distribution(random_engine);
            Point3d point(x, y, z);
            reference.push_back(point);

            double nx = noise_distribution(random_engine);
            double ny = noise_distribution(random_engine);
            double nz = noise_distribution(random_engine);
            Point3d noise(nx, ny, nz);

            auto transformed_point = transformation * point + noise * noise_level;
            transformed.push_back(transformed_point);
        }

        auto recovered_transformation = CalculateIcp(reference, transformed);

        EXPECT_LE((recovered_transformation.so3().matrix() - rotation).norm(), 1.0E-2);
        EXPECT_LE((recovered_transformation.translation() - translation).norm(), 1.0E-1);
    }
}

TEST(MathTest, RobustIcp) {
    std::default_random_engine random_engine(1234);
    std::uniform_real_distribution<double> uniform_distribution(-100.0, 100.0);
    std::uniform_real_distribution<double> inlier_noise_distribution(-1.0, 1.0);
    std::uniform_real_distribution<double> outlier_noise_distribution(20.0, 40.0);
    std::uniform_real_distribution<double> outlier_distribution(0.0, 1.0);

    const double outlier_fraction = 0.05;

    std::vector<Point3d> reference, transformed;
    Eigen::AngleAxisd angle_axis(0.84, Eigen::Vector3d(0.5, 0.3, 0.1).normalized());
    SE3d::SO3Type::Transformation rotation = angle_axis.toRotationMatrix();
    SE3d::TranslationType translation { 11.0, 23.0, 3.0 };
    SE3d transformation(rotation, translation);
    std::vector<uchar> inlier_mask;

    for (size_t index = 0; index < 100; ++index) {
        bool is_outlier = outlier_distribution(random_engine) < outlier_fraction;
        inlier_mask.push_back(is_outlier ? 0 : std::numeric_limits<uchar>::max());

        double x = uniform_distribution(random_engine);
        double y = uniform_distribution(random_engine);
        double z = uniform_distribution(random_engine);
        Point3d point(x, y, z);
        reference.push_back(point);

        const auto& noise_distribution = is_outlier ? outlier_noise_distribution : inlier_noise_distribution;
        double nx = noise_distribution(random_engine);
        double ny = noise_distribution(random_engine);
        double nz = noise_distribution(random_engine);
        Point3d noise(nx, ny, nz);

        auto transformed_point = transformation * point + noise;
        transformed.push_back(transformed_point);
    }

    auto recovered_transformation = CalculateIcp(reference, transformed);

    EXPECT_GE((recovered_transformation.so3().matrix() - rotation).norm(), 1.0E-2);
    EXPECT_GE((recovered_transformation.translation() - translation).norm(), 1.0E-1);

    SE3d robust_transformation;
    std::vector<uchar> estimated_inlier_mask;

    auto num_inliers = RobustIcp(reference, transformed, random_engine,
                                 robust_transformation, estimated_inlier_mask,
                                 30, 10, 3.0);

    EXPECT_LE((robust_transformation.so3().matrix() - rotation).norm(), 1.0E-2);
    EXPECT_LE((robust_transformation.translation() - translation).norm(), 0.5);

    for (size_t index = 0; index < inlier_mask.size(); ++index) {
        EXPECT_LE(inlier_mask[index], estimated_inlier_mask[index]);
    }
}