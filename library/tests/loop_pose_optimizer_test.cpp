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

#include "slammer/loop_pose_optimizer.h"

using namespace slammer;

namespace {

/// Generate a random transformation on SE(3)
///
/// The rotation will be chosen to be uniform, the translation will be normally distributed with
/// variance sigma^2.
SE3d GenerateRandomTransformation(double sigma = 1.0) {
    static std::random_device rd;

    std::uniform_real_distribution<> dis(-M_PI, M_PI);
    Vector3d axis = Eigen::Vector3d::Random().normalized();
    double angle = dis(rd);
    Quaterniond rotation(Eigen::AngleAxisd(angle, axis));

    std::normal_distribution<> norm(0.0, sigma);
    Vector3d translation;
    translation[0] = norm(rd);
    translation[1] = norm(rd);
    translation[2] = norm(rd);

    SE3d result = SE3d(rotation, translation);
    return result;
}

} // namespace

TEST(LoopPoseOptimizerTest, Derivatives) {
    SE3d T1 = GenerateRandomTransformation(),
        T2 = GenerateRandomTransformation();

    Eigen::Matrix<double, 6, 6> J;

    for (unsigned coord = 0; coord < 6; ++coord) {
        for (unsigned out = 0; out < 6; ++out) {
            SE3d T = T1 * T2;

            SE3d::Tangent tangent_out = T.log();
            SE3d::Tangent tangent_in = T1.log();

            double eps = 0.00001;

            SE3d::Tangent tangent = tangent_in;
            tangent[coord] += eps;
            SE3d T_e = SE3d::exp(tangent);

            SE3d::Tangent diff = (T_e * T2).log() - tangent_out;
            double slope = diff[out] / eps;
          
            J(out, coord) = slope;
        }
    }

    std::cout << J << std::endl << std::endl;

    Matrix3d denominator = (T1.so3() * T2.so3()).inverse().matrix();
    Matrix3d numerator = SE3d::SO3Type::hat(T1.so3().log()) * T1.so3().matrix() * T2.so3().matrix();

    Matrix3d J_rot = denominator * numerator;
    std::cout << J_rot << std::endl;
}