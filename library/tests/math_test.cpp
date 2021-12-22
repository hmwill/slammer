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

            auto transformed_point = transformation * point + noise * noise_level * 0.1;
            transformed.push_back(transformed_point);
        }

        auto recovered_transformation = CalculateIcp(reference, transformed);

        EXPECT_LE((recovered_transformation.so3().matrix() - rotation).norm(), 1.0E-2);
        EXPECT_LE((recovered_transformation.translation() - translation).norm(), 1.0E-2);
    }
}