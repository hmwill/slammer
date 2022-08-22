// BSD 3-Clause License
//
// Copyright (c) 2021, 2022, Hans-Martin Will
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

#include "slammer/pose_graph_optimizer.h"
#include "slammer/keyframe_index.h"

using namespace slammer;
using namespace Eigen;

// LoopPoseOptimizer_Derivatives_Test; this is a friend class to class PoseGraphOptimizer
//
// Purpose of this test is to verify that the partial derivatives calculated by 
// PoseGraphOptimizer::CalculateJacobianComponent are correct
TEST(PoseGraphOptimizer, Derivatives) {
    using PoseParameters = PoseGraphOptimizer::PoseParameters;
    using Constraints = Eigen::Matrix<double, PoseGraphOptimizer::kDimConstraint, 1>;

    // Approach:
    // 1. Calculate random before and after transformations
    // 2. Determine a set of parameters
    // 3. For each of the 6 dimensions in parameter space, calculate the upper and lower differential quotient
    // 4. Observe convergence as the interval of the quotient gets smaller
    // 5. The conerged value should match the corresponding value in the calculated Jacobian

    auto transform = 
        [](const SE3d& after, const PoseParameters& params, const SE3d& before) -> Constraints {
        auto delta = PoseGraphOptimizer::TransformFromParameters(params);
        auto total = after * delta * before;
        return total.matrix3x4().reshaped<Eigen::RowMajor>();
    };

    const std::vector<double> kEpsilon { /*1.0E-2,*/ 1.0E-4, /*1.0E-6*/ };

    auto test = [&](const SE3d& after, const PoseParameters& params, const SE3d& before) {
        // std::cout << "after: " << std::endl << after.matrix3x4() << std::endl;
        // std::cout << "before: " << std::endl << before.matrix3x4() << std::endl;

        // std::cout << "params:" << std::endl << params << std::endl;
        // std::cout << "delta transform: " << std::endl << PoseGraphOptimizer::TransformFromParameters(params).matrix3x4() << std::endl;
        // std::cout << "transform: " << std::endl << transform(after, params, before) << std::endl;
        

        auto J = PoseGraphOptimizer::CalculateJacobianComponent(after, params, before);
        // std::cout << "J: " << std::endl << J << std::endl;

        // run the test for each dimensions
        for (size_t dim = 0; dim < PoseGraphOptimizer::kDimPose; ++dim) {
            // run the test for each dimension of the transformation matrix
            PoseParameters delta = params;

            // std::cout << "dim: " << std::endl << dim << std::endl;

            for (auto eps: kEpsilon) {
                delta[dim] = params[dim] + eps;
                Constraints diff = transform(after, delta, before) - transform(after, params, before);
                Constraints quotient = diff * (1.0/eps);

                // std::cout << "eps: " << std::endl << eps << std::endl;
                // std::cout << "diff: " << std::endl << diff << std::endl;
                // std::cout << "quotient: " << std::endl << quotient << std::endl;

                Constraints div = J(all, dim) - quotient;
                // std::cout << "div: " << std::endl << div << std::endl;

                for (size_t index = 0; index < PoseGraphOptimizer::kDimConstraint; ++index) {
                    EXPECT_LE(fabs(quotient(index, 0) - J(index, dim)), 1.0E-3);
                }
            }
        }
    };

    // Parameters to drive a single tes case
    struct Parameters { 
        SE3d::Tangent after;
        PoseParameters params;
        SE3d::Tangent before;
    };

    const std::vector<Parameters> kTestParameters {
        {
            // Everything around identity
            { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 },
            { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 },
            { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 }
        },
        {
            // Delta translation
            { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 },
            { 0.0, 0.0, 0.0, 0.1, 0.2, 0.3 },
            { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 }
        },
        {
            // Delta rotation
            { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 },
            { 0.3, 0.2, 0.4, 0.0, 0.0, 0.0 },
            { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 }
        },

        // Adding before and after

        {
            // Delta rotation
            { 0.2, 0.3, 0.1, 0.5, 0.4, 0.3 },
            { 0.3, 0.2, 0.4, 0.0, 0.0, 0.0 },
            { 0.6, 0.4, 0.2, 0.2, 0.3, 0.1 }
        },
    };

    // Test driver loop
    for (const auto& test_set: kTestParameters) {
        SE3d after = SE3d::exp(test_set.after);
        SE3d before = SE3d::exp(test_set.before);

        test(after, test_set.params, before);        
    }
}

// This test case applies the optimizer to a configuration that already reflects the
// additional loop constraint. We therefore expect the resulting output poses to match the
// input poses.
TEST(PoseGraphOptimizer, DistortPoses) {
    // 1. Generate poses from noon to 9 hours along a circle
    std::vector<KeyframePointer> all_frames;

    constexpr double kTwoPi = M_PI * 2.0;
    constexpr double kScale = 10.0;
    constexpr size_t kLastHour = 9;
    constexpr auto kHours = 12.0;

    constexpr auto kDisturbTranslation = 0.5 / kTwoPi;
    constexpr auto kDisturbAngle = 0.25 / kTwoPi;

    double z = 1.0;

    for (size_t index = 0; index <= kLastHour; ++index, z = -z) {
        auto angle = index / kHours * kTwoPi;

        double x = sin(angle) * kScale;
        double y = cos(angle) * kScale;

        SE3d pose = SE3d::trans(x, y, z) * SE3d::rotZ(angle);

        KeyframePointer keyframe(new Keyframe());
        keyframe->pose = pose;
        keyframe->timestamp = Timestamp { Timediff { index } };

        all_frames.emplace_back(std::move(keyframe));
    }

    PoseGraphOptimizer::Keyframes keyframes;

    // include all but first frame into set to optimize
    for (size_t index = 1; index < all_frames.size(); ++index) {
        KeyframePointer& from = all_frames[index - 1];
        KeyframePointer& to = all_frames[index];

        // add constraints along circle
        from->covisible.insert(to);
        to->covisible.insert(from);

        // all to frames should be subject to optimization
        keyframes.push_back(to.get());
    }

    // 2. Add a new constraint connecting kLastHour to noon
    PoseGraphOptimizer::Constraints constraints {
        {
            all_frames[kLastHour].get(), all_frames[0].get(), 
            all_frames[0]->pose.inverse() * all_frames[kLastHour]->pose
        }
    };

    // 3. Create optimizer
    PoseGraphOptimizer optimizer { keyframes };
    PoseGraphOptimizer::Poses poses;

    // 3.1 Disturb the keyframe poses
    std::vector<SE3d> original, perturbed;

    std::mt19937_64 engine;
    std::uniform_real_distribution translate(0.0, kDisturbTranslation);
    std::uniform_real_distribution rotate(0.0, kDisturbAngle);

    std::for_each(keyframes.begin(), keyframes.end(), [&](Keyframe* keyframe){
        SE3d::Tangent distortion {
            translate(engine),
            translate(engine),
            translate(engine),
            rotate(engine),
            rotate(engine),
            rotate(engine),
        };

        original.push_back(keyframe->pose);
        perturbed.push_back(SE3d::exp(distortion) * keyframe->pose);
        poses.push_back(SE3d::exp(distortion) * keyframe->pose);
    });

    // 3.2 Run optimizer 
    PoseGraphOptimizer::Parameters parameters {
        40,
        0.01
    };

    Result<double> result = optimizer.Optimize(poses, true, constraints, parameters);

    EXPECT_TRUE(result.ok());
    // std::cout << "Residual error = " << result.value() << std::endl;

    // 4. We expect the output poses to more or less match the input poses, up to rounding error
    constexpr auto kEpsilon = 1.0E-3;
    
    for (size_t index = 0; index < keyframes.size(); ++index) {
        auto diffPose = poses[index].inverse() * original[index];
        auto delta = diffPose.log();

        // std::cout << "index:" << std::endl << index << std::endl;
        // std::cout << "original:" << std::endl << original[index].matrix3x4() << std::endl;
        // std::cout << "perturbed:" << std::endl << perturbed[index].matrix3x4() << std::endl;
        // std::cout << "reconstructed:" << std::endl << poses[index].matrix3x4() << std::endl;
        // std::cout << "diffPose:" << std::endl << diffPose.matrix3x4() << std::endl;
        // std::cout << "delta:" << std::endl << delta << std::endl;

        EXPECT_LE(delta.squaredNorm(), kEpsilon);
    }

    // Break graph cycles
    std::for_each(all_frames.begin(), all_frames.end(), [](auto& keyframe) {
        keyframe->covisible.clear();
    });
}

// This test case applies the optimizer to a configuration that already reflects the
// additional loop constraint. We therefore expect the resulting output poses to match the
// input poses.
//
// In this test, all edges are provided as explicit constraints
TEST(PoseGraphOptimizer, OnlyConstraints) {
    // 1. Generate poses from noon to 9 hours along a circle
    std::vector<KeyframePointer> all_frames;

    constexpr double kTwoPi = M_PI * 2.0;
    constexpr double kScale = 10.0;
    constexpr size_t kLastHour = 9;
    constexpr auto kHours = 12.0;

    constexpr auto kDisturbTranslation = 0.5 / kTwoPi;
    constexpr auto kDisturbAngle = 0.25 / kTwoPi;

    double z = 1.0;

    for (size_t index = 0; index <= kLastHour; ++index, z = -z) {
        auto angle = index / kHours * kTwoPi;

        double x = sin(angle) * kScale;
        double y = cos(angle) * kScale;

        SE3d pose = SE3d::trans(x, y, z) * SE3d::rotZ(angle);

        KeyframePointer keyframe(new Keyframe());
        keyframe->pose = pose;
        keyframe->timestamp = Timestamp { Timediff { index } };

        all_frames.emplace_back(std::move(keyframe));
    }

    PoseGraphOptimizer::Keyframes keyframes;

    // include all but first frame into set to optimize
    for (size_t index = 1; index < all_frames.size(); ++index) {
        // all to frames should be subject to optimization
        keyframes.push_back(all_frames[index].get());
    }

    // 2. Add a new constraint connecting kLastHour to noon
    PoseGraphOptimizer::Constraints constraints;
    
    for (size_t index = 0; index < all_frames.size(); ++index) {
        auto from = all_frames[index].get();
        auto to = all_frames[(index + 11) % all_frames.size()].get();

        constraints.emplace_back(PoseGraphOptimizer::Constraint {from, to, to->pose.inverse() * from->pose});
    }

    // 3. Create optimizer
    PoseGraphOptimizer optimizer { keyframes };
    PoseGraphOptimizer::Poses poses { keyframes.size() };

    // 3.1 Disturb the keyframe poses
    std::vector<SE3d> original, perturbed;

    std::mt19937_64 engine;
    std::uniform_real_distribution translate(0.0, kDisturbTranslation);
    std::uniform_real_distribution rotate(0.0, kDisturbAngle);

    std::for_each(keyframes.begin(), keyframes.end(), [&](Keyframe* keyframe){
        SE3d::Tangent distortion {
            translate(engine),
            translate(engine),
            translate(engine),
            rotate(engine),
            rotate(engine),
            rotate(engine),
        };

        original.push_back(keyframe->pose);
        perturbed.push_back(SE3d::exp(distortion) * keyframe->pose);
        keyframe->pose = SE3d::exp(distortion) * keyframe->pose;
    });

    // 3.2 Run optimizer 
    PoseGraphOptimizer::Parameters parameters {
        50,
        0.01
    };

    Result<double> result = optimizer.Optimize(poses, false, constraints, parameters);

    EXPECT_TRUE(result.ok());
    // std::cout << "Residual error = " << result.value() << std::endl;

    // 4. We expect the output poses to more or less match the input poses, up to rounding error
    constexpr auto kEpsilon = 1.0E-3;
    
    for (size_t index = 0; index < keyframes.size(); ++index) {
        auto diffPose = poses[index].inverse() * original[index];
        auto delta = diffPose.log();

        // std::cout << "index:" << std::endl << index << std::endl;
        // std::cout << "original:" << std::endl << original[index].matrix3x4() << std::endl;
        // std::cout << "perturbed:" << std::endl << perturbed[index].matrix3x4() << std::endl;
        // std::cout << "reconstructed:" << std::endl << poses[index].matrix3x4() << std::endl;
        // std::cout << "diffPose:" << std::endl << diffPose.matrix3x4() << std::endl;
        // std::cout << "delta:" << std::endl << delta << std::endl;

        EXPECT_LE(delta.squaredNorm(), kEpsilon);
    }

    // Break graph cycles
    std::for_each(all_frames.begin(), all_frames.end(), [](auto& keyframe) {
        keyframe->covisible.clear();
    });
}

