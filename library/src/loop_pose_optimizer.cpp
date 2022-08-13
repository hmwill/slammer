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

#include "slammer/loop_pose_optimizer.h"

#include "slammer/optimizer.h"

using namespace slammer;

using sparse::EmitTriplets;

void LoopPoseOptimizer::Initialize() {
    for (size_t index = 0; index < keyframes_.size() - 1; ++index) {
        SE3d relative_motion = keyframes_[index + 1]->pose.inverse() * keyframes_[index]->pose;
        measured_motion_.push_back(relative_motion);
    }
}

SE3d LoopPoseOptimizer::TransformFromParameters(const PoseParameters& params) {
    auto phi   = params[0];
    auto psi   = params[1];
    auto theta = params[2];
    auto x     = params[3];
    auto y     = params[4];
    auto z     = params[5];

    Eigen::Matrix4d T;

    T(0, 0) = cos(psi) * cos(theta);
    T(0, 1) = -cos(theta) * sin(psi);
    T(0, 2) = sin(theta);
    T(0, 3) = x;
    T(1, 0) = cos(phi) * sin(psi) + cos(psi) * sin(phi) * sin(theta);
    T(1, 1) = cos(phi) * cos(psi) - sin(phi) * sin(psi) * sin(theta);
    T(1, 2) = -cos(theta) * sin(phi);
    T(1, 3) = y;
    T(2, 0) = sin(phi) * sin(psi) - cos(phi) * cos(psi) * sin(theta);
    T(2, 1) = cos(psi) * sin(phi) + cos(phi) * sin(psi) * sin(theta);
    T(2, 2) = cos(phi) * cos(theta);
    T(2, 3) = z;
    T(3, 0) = 0.0;
    T(3, 1) = 0.0;
    T(3, 2) = 0.0;
    T(3, 3) = 1.0;

    return SE3d(T);
}

Eigen::Matrix<double, LoopPoseOptimizer::kDimConstraint, LoopPoseOptimizer::kDimPose> 
LoopPoseOptimizer::CalculateJacobianComponent(const SE3d& after, const PoseParameters& params, 
                                              const SE3d& before) {
    Eigen::Matrix<double, kDimConstraint, kDimPose> J;
    J.setZero();
    
    auto Ti = after.inverse().matrix3x4();
    auto Tj = before.matrix3x4();

    auto Ti1_1 = Ti(0, 0);
    auto Ti1_2 = Ti(0, 1);
    auto Ti1_3 = Ti(0, 2);
    auto Ti1_4 = Ti(0, 3);
    auto Ti2_1 = Ti(1, 0);
    auto Ti2_2 = Ti(1, 1);
    auto Ti2_3 = Ti(1, 2);
    auto Ti2_4 = Ti(1, 3);
    auto Ti3_1 = Ti(2, 0);
    auto Ti3_2 = Ti(2, 1);
    auto Ti3_3 = Ti(2, 2);
    auto Ti3_4 = Ti(2, 3);

    auto Tj1_1 = Tj(0, 0);
    auto Tj1_2 = Tj(0, 1);
    auto Tj1_3 = Tj(0, 2);
    auto Tj1_4 = Tj(0, 3);
    auto Tj2_1 = Tj(1, 0);
    auto Tj2_2 = Tj(1, 1);
    auto Tj2_3 = Tj(1, 2);
    auto Tj2_4 = Tj(1, 3);
    auto Tj3_1 = Tj(2, 0);
    auto Tj3_2 = Tj(2, 1);
    auto Tj3_3 = Tj(2, 2);
    auto Tj3_4 = Tj(2, 3);

    auto phi   = params[0];
    auto psi   = params[1];
    auto theta = params[2];
    auto x     = params[3];
    auto y     = params[4];
    auto z     = params[5];

    auto sin_phi = sin(phi);
    auto cos_phi = cos(phi);
    auto sin_psi = sin(psi);
    auto cos_psi = cos(psi);
    auto sin_theta = sin(theta);
    auto cos_theta = cos(theta);

    J(0, 0) = 
        - Tj1_1 * (Ti2_1 * (sin_phi * sin_psi - cos_phi * cos_psi * sin_theta) - 
                   Ti3_1 * (cos_phi * sin_psi + cos_psi * sin_phi * sin_theta)) 
        - Tj2_1 * (Ti2_1 * (cos_psi * sin_phi + cos_phi * sin_psi * sin_theta) - 
                   Ti3_1 * (cos_phi * cos_psi - sin_phi * sin_psi * sin_theta)) 
        - Tj3_1 * (Ti2_1 * cos_phi * cos_theta + Ti3_1 * cos_theta * sin_phi);
    J(0, 1) = 
        - Tj2_1 * (Ti2_1 * (cos_phi * sin_psi + cos_psi * sin_phi * sin_theta) + 
                   Ti3_1 * (sin_phi * sin_psi - cos_phi * cos_psi * sin_theta) + Ti1_1 * cos_psi * cos_theta) 
        + Tj1_1 * (Ti2_1 * (cos_phi * cos_psi - sin_phi * sin_psi * sin_theta) + 
                   Ti3_1 * (cos_psi * sin_phi + cos_phi * sin_psi * sin_theta) - Ti1_1 * cos_theta * sin_psi);
    J(0, 2) = 
          Tj2_1 * (Ti1_1 * sin_psi * sin_theta + Ti3_1 * cos_phi * cos_theta * sin_psi - 
                   Ti2_1 * cos_theta * sin_phi * sin_psi) + Tj3_1 * (Ti1_1 * cos_theta - 
                   Ti3_1 * cos_phi * sin_theta + Ti2_1 * sin_phi * sin_theta) 
        - Tj1_1 * (Ti1_1 * cos_psi * sin_theta + Ti3_1 * cos_phi * cos_psi * cos_theta - 
                   Ti2_1 * cos_psi * cos_theta * sin_phi);
    J(1, 0) = 
        - Tj1_2 * (Ti2_1 * (sin_phi * sin_psi - cos_phi * cos_psi * sin_theta) - 
                   Ti3_1 * (cos_phi * sin_psi + cos_psi * sin_phi * sin_theta)) 
        - Tj2_2 * (Ti2_1 * (cos_psi * sin_phi + cos_phi * sin_psi * sin_theta) - 
                   Ti3_1 * (cos_phi * cos_psi - sin_phi * sin_psi * sin_theta)) 
        - Tj3_2 * (Ti2_1 * cos_phi * cos_theta + Ti3_1 * cos_theta * sin_phi);
    J(1, 1) = 
        - Tj2_2 * (Ti2_1 * (cos_phi * sin_psi + cos_psi * sin_phi * sin_theta) + 
                   Ti3_1 * (sin_phi * sin_psi - cos_phi * cos_psi * sin_theta) + 
                   Ti1_1 * cos_psi * cos_theta) + 
                   Tj1_2 * (Ti2_1 * (cos_phi * cos_psi - sin_phi * sin_psi * sin_theta) + 
                   Ti3_1 * (cos_psi * sin_phi + cos_phi * sin_psi * sin_theta) - 
                   Ti1_1 * cos_theta * sin_psi);
    J(1, 2) = 
          Tj2_2 * (Ti1_1 * sin_psi * sin_theta + Ti3_1 * cos_phi * cos_theta * sin_psi - 
                   Ti2_1 * cos_theta * sin_phi * sin_psi) 
        + Tj3_2 * (Ti1_1 * cos_theta - Ti3_1 * cos_phi * sin_theta + Ti2_1 * sin_phi * sin_theta) 
        - Tj1_2 * (Ti1_1 * cos_psi * sin_theta + Ti3_1 * cos_phi * cos_psi * cos_theta - 
                   Ti2_1 * cos_psi * cos_theta * sin_phi);
    J(2, 0) = 
        - Tj1_3 * (Ti2_1 * (sin_phi * sin_psi - cos_phi * cos_psi * sin_theta) - 
                   Ti3_1 * (cos_phi * sin_psi + cos_psi * sin_phi * sin_theta)) 
        - Tj2_3 * (Ti2_1 * (cos_psi * sin_phi + cos_phi * sin_psi * sin_theta) - 
                   Ti3_1 * (cos_phi * cos_psi - sin_phi * sin_psi * sin_theta)) 
        - Tj3_3 * (Ti2_1 * cos_phi * cos_theta + Ti3_1 * cos_theta * sin_phi);
    J(2, 1) = 
        - Tj2_3 * (Ti2_1 * (cos_phi * sin_psi + cos_psi * sin_phi * sin_theta) + 
                   Ti3_1 * (sin_phi * sin_psi - cos_phi * cos_psi * sin_theta) + 
                   Ti1_1 * cos_psi * cos_theta) 
        + Tj1_3 * (Ti2_1 * (cos_phi * cos_psi - sin_phi * sin_psi * sin_theta) + 
                   Ti3_1 * (cos_psi * sin_phi + cos_phi * sin_psi * sin_theta) - 
                   Ti1_1 * cos_theta * sin_psi);
    J(2, 2) = 
          Tj2_3 * (Ti1_1 * sin_psi * sin_theta + Ti3_1 * cos_phi * cos_theta * sin_psi - 
                   Ti2_1 * cos_theta * sin_phi * sin_psi) 
        + Tj3_3 * (Ti1_1 * cos_theta - Ti3_1 * cos_phi * sin_theta + Ti2_1 * sin_phi * sin_theta) 
        - Tj1_3 * (Ti1_1 * cos_psi * sin_theta + Ti3_1 * cos_phi * cos_psi * cos_theta - 
                   Ti2_1 * cos_psi * cos_theta * sin_phi);
    J(3, 0) = 
        - Tj1_4 * (Ti2_1 * (sin_phi * sin_psi - cos_phi * cos_psi * sin_theta) - 
                   Ti3_1 * (cos_phi * sin_psi + cos_psi * sin_phi * sin_theta)) 
        - Tj2_4 * (Ti2_1 * (cos_psi * sin_phi + cos_phi * sin_psi * sin_theta) - 
                   Ti3_1 * (cos_phi * cos_psi - sin_phi * sin_psi * sin_theta)) 
        - Tj3_4 * (Ti2_1 * cos_phi * cos_theta + Ti3_1 * cos_theta * sin_phi);
    J(3, 1) = 
        - Tj2_4 * (Ti2_1 * (cos_phi * sin_psi + cos_psi * sin_phi * sin_theta) + 
                   Ti3_1 * (sin_phi * sin_psi - cos_phi * cos_psi * sin_theta) + 
                   Ti1_1 * cos_psi * cos_theta) 
        + Tj1_4 * (Ti2_1 * (cos_phi * cos_psi - sin_phi * sin_psi * sin_theta) + 
                   Ti3_1 * (cos_psi * sin_phi + cos_phi * sin_psi * sin_theta) - 
                   Ti1_1 * cos_theta * sin_psi);
    J(3, 2) = 
          Tj2_4 * (Ti1_1 * sin_psi * sin_theta + Ti3_1 * cos_phi * cos_theta * sin_psi - 
                   Ti2_1 * cos_theta * sin_phi * sin_psi) 
        + Tj3_4 * (Ti1_1 * cos_theta - Ti3_1 * cos_phi * sin_theta + Ti2_1 * sin_phi * sin_theta) 
        - Tj1_4 * (Ti1_1 * cos_psi * sin_theta + 
                   Ti3_1 * cos_phi * cos_psi * cos_theta - 
                   Ti2_1 * cos_psi * cos_theta * sin_phi);
    J(3, 3) = Ti1_1;
    J(3, 4) = Ti2_1;
    J(3, 5) = Ti3_1;
    J(4, 0) = 
        - Tj1_1 * (Ti2_2 * (sin_phi * sin_psi - cos_phi * cos_psi * sin_theta) - 
                   Ti3_2 * (cos_phi * sin_psi + cos_psi * sin_phi * sin_theta)) 
        - Tj2_1 * (Ti2_2 * (cos_psi * sin_phi + cos_phi * sin_psi * sin_theta) - 
                   Ti3_2 * (cos_phi * cos_psi - sin_phi * sin_psi * sin_theta)) 
        - Tj3_1 * (Ti2_2 * cos_phi * cos_theta + Ti3_2 * cos_theta * sin_phi);
    J(4, 1) = 
        - Tj2_1 * (Ti2_2 * (cos_phi * sin_psi + cos_psi * sin_phi * sin_theta) + 
                   Ti3_2 * (sin_phi * sin_psi - cos_phi * cos_psi * sin_theta) + 
                   Ti1_2 * cos_psi * cos_theta) 
        + Tj1_1 * (Ti2_2 * (cos_phi * cos_psi - sin_phi * sin_psi * sin_theta) + 
                   Ti3_2 * (cos_psi * sin_phi + cos_phi * sin_psi * sin_theta) - 
                   Ti1_2 * cos_theta * sin_psi);
    J(4, 2) = 
          Tj2_1 * (Ti1_2 * sin_psi * sin_theta + 
                   Ti3_2 * cos_phi * cos_theta * sin_psi - 
                   Ti2_2 * cos_theta * sin_phi * sin_psi) 
        + Tj3_1 * (Ti1_2 * cos_theta - 
                   Ti3_2 * cos_phi * sin_theta + 
                   Ti2_2 * sin_phi * sin_theta) 
        - Tj1_1 * (Ti1_2 * cos_psi * sin_theta + 
                   Ti3_2 * cos_phi * cos_psi * cos_theta - 
                   Ti2_2 * cos_psi * cos_theta * sin_phi);
    J(5, 0) = 
        - Tj1_2 * (Ti2_2 * (sin_phi * sin_psi - cos_phi * cos_psi * sin_theta) - 
                   Ti3_2 * (cos_phi * sin_psi + cos_psi * sin_phi * sin_theta)) 
        - Tj2_2 * (Ti2_2 * (cos_psi * sin_phi + cos_phi * sin_psi * sin_theta) - 
                   Ti3_2 * (cos_phi * cos_psi - sin_phi * sin_psi * sin_theta)) 
        - Tj3_2 * (Ti2_2 * cos_phi * cos_theta + Ti3_2 * cos_theta * sin_phi);
    J(5, 1) = 
        - Tj2_2 * (Ti2_2 * (cos_phi * sin_psi + cos_psi * sin_phi * sin_theta) + 
                   Ti3_2 * (sin_phi * sin_psi - cos_phi * cos_psi * sin_theta) + 
                   Ti1_2 * cos_psi * cos_theta) 
        + Tj1_2 * (Ti2_2 * (cos_phi * cos_psi - sin_phi * sin_psi * sin_theta) + 
                   Ti3_2 * (cos_psi * sin_phi + cos_phi * sin_psi * sin_theta) - 
                   Ti1_2 * cos_theta * sin_psi);
    J(5, 2) = 
          Tj2_2 * (Ti1_2 * sin_psi * sin_theta + 
                   Ti3_2 * cos_phi * cos_theta * sin_psi - 
                   Ti2_2 * cos_theta * sin_phi * sin_psi) 
        + Tj3_2 * (Ti1_2 * cos_theta - 
                   Ti3_2 * cos_phi * sin_theta + 
                   Ti2_2 * sin_phi * sin_theta) 
        - Tj1_2 * (Ti1_2 * cos_psi * sin_theta + 
                   Ti3_2 * cos_phi * cos_psi * cos_theta - 
                   Ti2_2 * cos_psi * cos_theta * sin_phi);
    J(6, 0) = 
        - Tj1_3 * (Ti2_2 * (sin_phi * sin_psi - cos_phi * cos_psi * sin_theta) - 
                   Ti3_2 * (cos_phi * sin_psi + cos_psi * sin_phi * sin_theta)) 
        - Tj2_3 * (Ti2_2 * (cos_psi * sin_phi + cos_phi * sin_psi * sin_theta) - 
                   Ti3_2 * (cos_phi * cos_psi - sin_phi * sin_psi * sin_theta)) 
        - Tj3_3 * (Ti2_2 * cos_phi * cos_theta + Ti3_2 * cos_theta * sin_phi);
    J(6, 1) = 
        - Tj2_3 * (Ti2_2 * (cos_phi * sin_psi + cos_psi * sin_phi * sin_theta) + 
                   Ti3_2 * (sin_phi * sin_psi - cos_phi * cos_psi * sin_theta) + 
                   Ti1_2 * cos_psi * cos_theta) 
        + Tj1_3 * (Ti2_2 * (cos_phi * cos_psi - sin_phi * sin_psi * sin_theta) + 
                   Ti3_2 * (cos_psi * sin_phi + cos_phi * sin_psi * sin_theta) - 
                   Ti1_2 * cos_theta * sin_psi);
    J(6, 2) = 
          Tj2_3 * (Ti1_2 * sin_psi * sin_theta + 
                   Ti3_2 * cos_phi * cos_theta * sin_psi - 
                   Ti2_2 * cos_theta * sin_phi * sin_psi) 
        + Tj3_3 * (Ti1_2 * cos_theta - 
                   Ti3_2 * cos_phi * sin_theta + 
                   Ti2_2 * sin_phi * sin_theta) 
        - Tj1_3 * (Ti1_2 * cos_psi * sin_theta + 
                   Ti3_2 * cos_phi * cos_psi * cos_theta - 
                   Ti2_2 * cos_psi * cos_theta * sin_phi);
    J(7, 0) = 
        - Tj1_4 * (Ti2_2 * (sin_phi * sin_psi - cos_phi * cos_psi * sin_theta) - 
                   Ti3_2 * (cos_phi * sin_psi + cos_psi * sin_phi * sin_theta)) 
        - Tj2_4 * (Ti2_2 * (cos_psi * sin_phi + cos_phi * sin_psi * sin_theta) - 
                   Ti3_2 * (cos_phi * cos_psi - sin_phi * sin_psi * sin_theta)) 
        - Tj3_4 * (Ti2_2 * cos_phi * cos_theta + Ti3_2 * cos_theta * sin_phi);
    J(7, 1) = 
        - Tj2_4 * (Ti2_2 * (cos_phi * sin_psi + cos_psi * sin_phi * sin_theta) + 
                   Ti3_2 * (sin_phi * sin_psi - cos_phi * cos_psi * sin_theta) + 
                   Ti1_2 * cos_psi * cos_theta) 
        + Tj1_4 * (Ti2_2 * (cos_phi * cos_psi - sin_phi * sin_psi * sin_theta) + 
                   Ti3_2 * (cos_psi * sin_phi + cos_phi * sin_psi * sin_theta) - 
                   Ti1_2 * cos_theta * sin_psi);
    J(7, 2) = 
          Tj2_4 * (Ti1_2 * sin_psi * sin_theta + 
                   Ti3_2 * cos_phi * cos_theta * sin_psi - 
                   Ti2_2 * cos_theta * sin_phi * sin_psi) 
        + Tj3_4 * (Ti1_2 * cos_theta - 
                   Ti3_2 * cos_phi * sin_theta + 
                   Ti2_2 * sin_phi * sin_theta) 
        - Tj1_4 * (Ti1_2 * cos_psi * sin_theta + 
                   Ti3_2 * cos_phi * cos_psi * cos_theta - 
                   Ti2_2 * cos_psi * cos_theta * sin_phi);
    J(7, 3) = Ti1_2;
    J(7, 4) = Ti2_2;
    J(7, 5) = Ti3_2;
    J(8, 0) = 
        - Tj1_1 * (Ti2_3 * (sin_phi * sin_psi - cos_phi * cos_psi * sin_theta) - 
                   Ti3_3 * (cos_phi * sin_psi + cos_psi * sin_phi * sin_theta)) 
        - Tj2_1 * (Ti2_3 * (cos_psi * sin_phi + cos_phi * sin_psi * sin_theta) - 
                   Ti3_3 * (cos_phi * cos_psi - sin_phi * sin_psi * sin_theta)) 
        - Tj3_1 * (Ti2_3 * cos_phi * cos_theta + 
                   Ti3_3 * cos_theta * sin_phi);
    J(8, 1) = 
        - Tj2_1 * (Ti2_3 * (cos_phi * sin_psi + cos_psi * sin_phi * sin_theta) + 
                   Ti3_3 * (sin_phi * sin_psi - cos_phi * cos_psi * sin_theta) + 
                   Ti1_3 * cos_psi * cos_theta) 
        + Tj1_1 * (Ti2_3 * (cos_phi * cos_psi - sin_phi * sin_psi * sin_theta) + 
                   Ti3_3 * (cos_psi * sin_phi + cos_phi * sin_psi * sin_theta) - 
                   Ti1_3 * cos_theta * sin_psi);
    J(8, 2) = 
          Tj2_1 * (Ti1_3 * sin_psi * sin_theta + 
                   Ti3_3 * cos_phi * cos_theta * sin_psi - 
                   Ti2_3 * cos_theta * sin_phi * sin_psi) 
        + Tj3_1 * (Ti1_3 * cos_theta - 
                   Ti3_3 * cos_phi * sin_theta + 
                   Ti2_3 * sin_phi * sin_theta) 
        - Tj1_1 * (Ti1_3 * cos_psi * sin_theta + 
                   Ti3_3 * cos_phi * cos_psi * cos_theta - 
                   Ti2_3 * cos_psi * cos_theta * sin_phi);
    J(9, 0) = 
        - Tj1_2 * (Ti2_3 * (sin_phi * sin_psi - cos_phi * cos_psi * sin_theta) - 
                   Ti3_3 * (cos_phi * sin_psi + cos_psi * sin_phi * sin_theta)) 
        - Tj2_2 * (Ti2_3 * (cos_psi * sin_phi + cos_phi * sin_psi * sin_theta) - 
                   Ti3_3 * (cos_phi * cos_psi - sin_phi * sin_psi * sin_theta)) 
        - Tj3_2 * (Ti2_3 * cos_phi * cos_theta + Ti3_3 * cos_theta * sin_phi);
    J(9, 1) = 
        - Tj2_2 * (Ti2_3 * (cos_phi * sin_psi + cos_psi * sin_phi * sin_theta) + 
                   Ti3_3 * (sin_phi * sin_psi - cos_phi * cos_psi * sin_theta) + 
                   Ti1_3 * cos_psi * cos_theta) 
        + Tj1_2 * (Ti2_3 * (cos_phi * cos_psi - sin_phi * sin_psi * sin_theta) + 
                   Ti3_3 * (cos_psi * sin_phi + cos_phi * sin_psi * sin_theta) - 
                   Ti1_3 * cos_theta * sin_psi);
    J(9, 2) = 
          Tj2_2 * (Ti1_3 * sin_psi * sin_theta + 
                   Ti3_3 * cos_phi * cos_theta * sin_psi - 
                   Ti2_3 * cos_theta * sin_phi * sin_psi) 
        + Tj3_2 * (Ti1_3 * cos_theta - 
                   Ti3_3 * cos_phi * sin_theta + 
                   Ti2_3 * sin_phi * sin_theta) 
        - Tj1_2 * (Ti1_3 * cos_psi * sin_theta + 
                   Ti3_3 * cos_phi * cos_psi * cos_theta - 
                   Ti2_3 * cos_psi * cos_theta * sin_phi);
    J(10, 0) = 
        - Tj1_3 * (Ti2_3 * (sin_phi * sin_psi - cos_phi * cos_psi * sin_theta) - 
                   Ti3_3 * (cos_phi * sin_psi + cos_psi * sin_phi * sin_theta)) 
        - Tj2_3 * (Ti2_3 * (cos_psi * sin_phi + cos_phi * sin_psi * sin_theta) - 
                   Ti3_3 * (cos_phi * cos_psi - sin_phi * sin_psi * sin_theta)) 
        - Tj3_3 * (Ti2_3 * cos_phi * cos_theta + Ti3_3 * cos_theta * sin_phi);
    J(10, 1) = 
        - Tj2_3 * (Ti2_3 * (cos_phi * sin_psi + cos_psi * sin_phi * sin_theta) + 
                   Ti3_3 * (sin_phi * sin_psi - cos_phi * cos_psi * sin_theta) + 
                   Ti1_3 * cos_psi * cos_theta) 
        + Tj1_3 * (Ti2_3 * (cos_phi * cos_psi - sin_phi * sin_psi * sin_theta) + 
                   Ti3_3 * (cos_psi * sin_phi + cos_phi * sin_psi * sin_theta) - 
                   Ti1_3 * cos_theta * sin_psi);
    J(10, 2) = 
          Tj2_3 * (Ti1_3 * sin_psi * sin_theta + 
                   Ti3_3 * cos_phi * cos_theta * sin_psi - 
                   Ti2_3 * cos_theta * sin_phi * sin_psi) 
        + Tj3_3 * (Ti1_3 * cos_theta - Ti3_3 * cos_phi * sin_theta + Ti2_3 * sin_phi * sin_theta) 
        - Tj1_3 * (Ti1_3 * cos_psi * sin_theta + 
                   Ti3_3 * cos_phi * cos_psi * cos_theta - 
                   Ti2_3 * cos_psi * cos_theta * sin_phi);
    J(11, 0) = 
        - Tj1_4 * (Ti2_3 * (sin_phi * sin_psi - cos_phi * cos_psi * sin_theta) - 
                   Ti3_3 * (cos_phi * sin_psi + cos_psi * sin_phi * sin_theta)) 
        - Tj2_4 * (Ti2_3 * (cos_psi * sin_phi + cos_phi * sin_psi * sin_theta) - 
                   Ti3_3 * (cos_phi * cos_psi - sin_phi * sin_psi * sin_theta)) 
        - Tj3_4 * (Ti2_3 * cos_phi * cos_theta + Ti3_3 * cos_theta * sin_phi);
    J(11, 1) = 
        - Tj2_4 * (Ti2_3 * (cos_phi * sin_psi + cos_psi * sin_phi * sin_theta) + 
                   Ti3_3 * (sin_phi * sin_psi - cos_phi * cos_psi * sin_theta) + 
                   Ti1_3 * cos_psi * cos_theta) 
        + Tj1_4 * (Ti2_3 * (cos_phi * cos_psi - sin_phi * sin_psi * sin_theta) + 
                   Ti3_3 * (cos_psi * sin_phi + cos_phi * sin_psi * sin_theta) - 
                   Ti1_3 * cos_theta * sin_psi);
    J(11, 2) = 
          Tj2_4 * (Ti1_3 * sin_psi * sin_theta + 
                   Ti3_3 * cos_phi * cos_theta * sin_psi - 
                   Ti2_3 * cos_theta * sin_phi * sin_psi) 
        + Tj3_4 * (Ti1_3 * cos_theta - 
                   Ti3_3 * cos_phi * sin_theta + 
                   Ti2_3 * sin_phi * sin_theta) 
        - Tj1_4 * (Ti1_3 * cos_psi * sin_theta + 
                   Ti3_3 * cos_phi * cos_psi * cos_theta - 
                   Ti2_3 * cos_psi * cos_theta * sin_phi);
    J(11, 3) = Ti1_3;
    J(11, 4) = Ti2_3;
    J(11, 5) = Ti3_3;

    return J;
}

Result<double> LoopPoseOptimizer::Optimize(Poses &poses, bool inout, SE3d relative_motion,
                                           const Parameters &parameters) {
    assert(poses.size() == keyframes_.size());

    // initialize all correction values to 0
    Eigen::VectorXd value(total_dimension());
    value.setZero();

    // Initialize pose vector from keyframe poses in case inout isn't set
    if (!inout) {
        for (size_t index = 0; index < poses.size(); ++index) {
            poses[index] = keyframes_[index]->pose;
        }
    }

    // perform optimization
    using namespace std::placeholders;

    auto result =
        LevenbergMarquardt(std::bind(&LoopPoseOptimizer::CalculateJacobian, *this, poses, relative_motion, _1),
                           std::bind(&LoopPoseOptimizer::CalculateResidual, *this, poses, relative_motion, _1),
                           value, parameters.max_iterations, parameters.lambda);

    // extract result: multiply each pose with the delta determined through the optimization process
    for (size_t index = 1; index < poses.size(); ++index) {
        auto slot = PoseSlot(index);
        auto delta = TransformFromParameters(value(slot));
        poses[index] = delta * poses[index];
    }

    return result;
}

Eigen::SparseMatrix<double>
LoopPoseOptimizer::CalculateJacobian(const Poses &poses, const SE3d &relative_motion,
                                     const Eigen::VectorXd &value) const {
    using Triplet = Eigen::Triplet<double>;
    std::vector<Triplet> triplets;
    size_t num_constraints = 0;

    // Special case: edge from 0 to 1 (frame 0 is fixed)
    // Calculate Jacobian for to node
    auto J_to = -CalculateJacobianComponent(poses[0].inverse(), value(PoseSlot(1)), poses[1]);
    EmitTriplets(J_to, std::back_insert_iterator(triplets), 0, 0);
    num_constraints += kDimConstraint;

    for (size_t index = 1; index < keyframes_.size() - 1; ++index, num_constraints += kDimConstraint) {
        size_t from_index = index;
        size_t to_index = index + 1;

        // Calculate Jacobian for from node
        auto J_from = CalculateJacobianComponent(poses[to_index].inverse(), value(PoseSlot(from_index)), 
                                                 poses[from_index]);
        EmitTriplets(J_from, std::back_insert_iterator(triplets), index * kDimConstraint, 
                     (from_index - 1) * kDimPose);

        // Calculate Jacobian for to node
        auto J_to = -CalculateJacobianComponent(poses[from_index].inverse(), value(PoseSlot(to_index)), 
                                                poses[to_index]);
        EmitTriplets(J_to, std::back_insert_iterator(triplets), index * kDimConstraint, (to_index - 1) * kDimPose);
    }

    // generate closing constraint from last keyframe to first
    size_t last_pose_index = keyframes_.size() - 1;
    auto J_from = CalculateJacobianComponent(poses[0].inverse(), value(PoseSlot(last_pose_index)), 
                                             poses[last_pose_index]);
    EmitTriplets(J_from, std::back_insert_iterator(triplets), num_constraints, (last_pose_index - 1) * kDimPose);
    num_constraints += kDimConstraint;
    assert(num_constraints == total_constraints());

    Eigen::SparseMatrix<double> result(total_constraints(), total_dimension());
    result.setFromTriplets(triplets.begin(), triplets.end());
    return result;
}

void LoopPoseOptimizer::CalculateResidual0(const Poses &poses, Eigen::VectorXd &residual, const SE3d &measured_motion,
                                           const Eigen::VectorXd &value, size_t from_index, size_t to_index,
                                           size_t residual_index) const {
    auto residual_slot = Eigen::seqN(residual_index * kDimConstraint, int(kDimConstraint));

    auto from_pose = 
        from_index > 0 ? TransformFromParameters(value(PoseSlot(from_index))) * poses[from_index] : poses[from_index];

    auto to_pose = 
        to_index > 0 ? TransformFromParameters(value(PoseSlot(to_index))) * poses[to_index] : poses[to_index];

    auto diff_motion = (to_pose.inverse() * from_pose).matrix3x4() - measured_motion.matrix3x4();
    residual(residual_slot) = diff_motion.reshaped<Eigen::RowMajor>();
}

// The residual is comprised of blocks of six row each, in order of the provided keyframes. The six rows correspond
// to the 6 dimensions of the logarithm of T_ij^-1 * T_j * T_i^-1, where j = (i + 1) mod N, N the number of
// keyframes.
Eigen::VectorXd
LoopPoseOptimizer::CalculateResidual(const Poses &poses, const SE3d &relative_motion, 
                                     const Eigen::VectorXd &value) const {
    Eigen::VectorXd residual(total_constraints());

    size_t residual_index = 0;

    // determine the residual for subsequent pairs of keyframes along the linear sequence
    for (size_t index = 0; index < keyframes_.size() - 1; ++index) {
        CalculateResidual0(poses, residual, measured_motion_[index], value, index, index + 1, index);
    }

    // determine the residual for the new constraint between last and first keyframe that we are adding in order
    // to close the loop
    CalculateResidual0(poses, residual, relative_motion, value, keyframes_.size() - 1, 0, keyframes_.size() - 1);

    return residual;
}
