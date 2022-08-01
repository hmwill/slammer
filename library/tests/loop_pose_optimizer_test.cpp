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
#include "slammer/keyframe_index.h"

using namespace slammer;

TEST(LoopPoseOptimizer, Circle) {
    // 1. Generate poses from noon to 10 hours along a circle
    LoopPoseOptimizer::Keyframes keyframes;

    const double kTwoPi = M_PI * 2.0;
    const double kScale = 10.0;

    for (size_t index = 0; index <= 10; ++index) {
        auto angle = index / 12.0 * kTwoPi;

        double x = sin(angle) * kScale;
        double y = cos(angle) * kScale;
        double z = 0;

        SE3d pose = SE3d::trans(x, y, z) * SE3d::rotZ(angle);

        KeyframePointer keyframe(new Keyframe());
        keyframe->pose = pose;
        keyframes.push_back(keyframe);
    }

    // 2. Add a new constraint connecting 10 to noon
    SE3d relative_motion = keyframes[1]->pose * keyframes[0]->pose.inverse();

    // 3. Run optimizer
    LoopPoseOptimizer optimizer { keyframes };
    LoopPoseOptimizer::Poses poses(keyframes.size());

    LoopPoseOptimizer::Parameters parameters {
        200,
        0.1
    };

    Result<double> result = optimizer.Optimize(poses, false, relative_motion, parameters);

    // 4. the resulting poses should be appromimately around a circle

    // Keyframe 0 should still be where it was before
    auto diff = poses[0].inverse() * keyframes[0]->pose;
    ASSERT_DOUBLE_EQ(diff.translation().squaredNorm(), 0.0);
    ASSERT_DOUBLE_EQ((diff.rotationMatrix() - Matrix3d::Identity(3, 3)).squaredNorm(), 0.0);

    // the center of gravity of all pose translations should still be approximately at x = 0, z = 0
    Vector3d center = Vector3d::Zero();
    for (auto iter = poses.begin(); iter != poses.end(); ++iter) {
        center += iter->translation();
    }

    center *= 1.0 / poses.size();
    //ASSERT_LE(fabs(center.x()), 1.0E-6);
    //ASSERT_DOUBLE_EQ(fabs(center.z()), 0.0);

    // the distance of all pose translations from the center of gravity should be roughly the same

    // 5. we'd expect an approximately even spacing
}