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
        SE3d relative_motion = keyframes_[index + 1]->pose * keyframes_[index]->pose.inverse();
        relative_motion_.push_back(relative_motion);
    }
}

Result<double> LoopPoseOptimizer::Optimize(Poses& poses, bool inout, SE3d relative_motion, 
                                           const Parameters& parameters) {
    assert(poses.size() == keyframes_.size());

    Eigen::VectorXd value(total_dimension());

    // Initialize value vector
    if (inout) {
        for (size_t index = 0; index < poses.size(); ++index) {
            auto slot = PoseSlot(index);
            value(slot) = poses[index].log();
        }
    } else {
        for (size_t index = 0; index < keyframes_.size(); ++index) {
            auto slot = PoseSlot(index);
            value(slot) = keyframes_[index]->pose.log();
        }
    }

    // perform optimization
    using namespace std::placeholders;

    auto result = 
        LevenbergMarquardt(std::bind(&LoopPoseOptimizer::CalculateJacobian, *this, relative_motion, _1),
                            std::bind(&LoopPoseOptimizer::CalculateResidual, *this, relative_motion, _1),
                            value, parameters.max_iterations, parameters.lambda);

    // extract result
    for (size_t index = 0; index < poses.size(); ++index) {
        auto slot = PoseSlot(index);
        poses[index] = SE3d::exp(value(slot));
    }

    return result;
}

// The rows of the Jacobian are blocks of six rows each, in the order of the provided keyframes. The six rows correspond
// to the 6 dimensions of the logarithm of T_ij^-1 * T_j * T_i^-1, where j = (i + 1) mod N, N the number of
// keyframes.

Eigen::SparseMatrix<double> 
LoopPoseOptimizer::CalculateJacobian(const SE3d& relative_motion, const Eigen::VectorXd& value) const {
    using Triplet = Eigen::Triplet<double>;
    std::vector<Triplet> triplets;
    size_t num_constraints = 0;

    // TODO: Generate matrix
    for (size_t index = 0; index < keyframes_.size() - 1; ++index, num_constraints += kDimConstraint) {
        size_t from_index = index;
        size_t to_index = index + 1;
        const auto& measured_motion = relative_motion_[index];

        auto from_slot = PoseSlot(from_index);
        auto to_slot = PoseSlot(to_index + 1);
        auto from_pose = SE3d::exp(value(from_slot));
        auto to_pose = SE3d::exp(value(to_slot));
        auto counter_motion = to_pose.inverse() * measured_motion;
        auto diff_motion = from_pose * counter_motion;



        // Eigen::Matrix<double, 3, 6> jacobian_log;
        // jacobian_log(Eigen::seqN(0, 3), Eigen::seqN(0, 3)) = Matrix3d::Identity();
        // jacobian_log(Eigen::seqN(0, 3), Eigen::seqN(3, 3)) = -SE3d::SO3Member::hat(transformed);
        // Eigen::Matrix<double, 3, 6> jacobian_params = jacobian_project * jacobian_second_params;
    }

    // generate closing constraint from last keyframe to first


    num_constraints += kDimConstraint;
    assert(num_constraints == total_constraints());

    Eigen::SparseMatrix<double> result(total_constraints(), total_dimension());
    result.setFromTriplets(triplets.begin(), triplets.end());
    return result;
}

void LoopPoseOptimizer::CalculateResidual0(Eigen::VectorXd& residual, const SE3d& relative_motion, const Eigen::VectorXd& value,
    size_t from_index, size_t to_index, size_t residual_index) const {
    auto residual_slot = Eigen::seqN(residual_index * kDimConstraint, int(kDimConstraint));
    auto from_slot = PoseSlot(from_index);
    auto to_slot = PoseSlot(to_index + 1);
    auto from_pose = SE3d::exp(value(from_slot));
    auto to_pose = SE3d::exp(value(to_slot));
    auto diff_motion = from_pose * to_pose.inverse() * relative_motion;
    residual(residual_slot) = diff_motion.log();
}

// The residual is comprised of blocks of six row each, in order of the provided keyframes. The six rows correspond
// to the 6 dimensions of the logarithm of T_ij^-1 * T_j * T_i^-1, where j = (i + 1) mod N, N the number of
// keyframes.
Eigen::VectorXd 
LoopPoseOptimizer::CalculateResidual(const SE3d& relative_motion, const Eigen::VectorXd& value) const {
    Eigen::VectorXd residual(total_constraints());
  
    size_t residual_index = 0;

    // determine the residual for subsequent pairs of keyframes along the linear sequence
    for (size_t index = 0; index < keyframes_.size() - 1; ++index) {
        CalculateResidual0(residual, relative_motion_[index], value, index, index + 1, index);
    }

    // determine the residual for the new constraint between last and first keyframe that we are adding in order
    // to close the loop
    CalculateResidual0(residual, relative_motion, value, keyframes_.size() - 1, 0, keyframes_.size() - 1);

    return residual;
}
