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
    // 1. Generate poses from noon to 9 hours along a circle
    LoopPoseOptimizer::Keyframes keyframes;

    const double kTwoPi = M_PI * 2.0;
    const double kScale = 10.0;

    for (size_t index = 0; index <= 9; ++index) {
        auto angle = index / 12.0 * kTwoPi;

        double x = sin(angle) * kScale;
        double y = cos(angle) * kScale;
        double z = 0;

        SE3d pose;
        pose.trans(x, y, z);
        pose.rotZ(angle);

        KeyframePointer keyframe(new Keyframe());
        keyframe->pose = pose;
        keyframes.push_back(keyframe);
    }

    // 2. Add a new constraint connecting 9 to noon
    SE3d relative_motion = keyframes[1]->pose * keyframes[0]->pose.inverse();

    // 3. Run optimizer
    LoopPoseOptimizer optimizer { keyframes };
    LoopPoseOptimizer::Poses poses(keyframes.size());

    LoopPoseOptimizer::Parameters parameters {
        10,
        0.1
    };

    Result<double> result = optimizer.Optimize(poses, false, relative_motion, parameters);

    // 4. the resulting poses should be appromimately around a circle


    // 5. we'd expect an approximately even spacing
}