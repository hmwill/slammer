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

#ifndef SLAMMER_LOOP_POSE_OPTIMIZER_H
#define SLAMMER_LOOP_POSE_OPTIMIZER_H

#pragma once

#include "slammer/slammer.h"
#include "slammer/map.h"

#include "Eigen/Sparse"

namespace slammer {

class LoopPoseOptimizer {
public:
    struct Parameters {
        /// maximum number of iterations
        unsigned max_iterations;

        /// the lambda parameter for Levenberg-Marquardt
        double lambda;
    };

    using Keyframes = std::vector<KeyframePointer>;
    using Poses = std::vector<SE3d>;

    /// Construct and instance of the pose graph optimization problem
    ///
    /// \param keyframes    the keyframes included in the subgraph that are subject to optimization
    LoopPoseOptimizer(const Keyframes& keyframes)
        : keyframes_(keyframes) {
        Initialize();
    }

    /// Construct and instance of the pose graph optimization problem
    ///
    /// \param keyframes    the keyframes included in the subgraph that are subject to optimization
    LoopPoseOptimizer(Keyframes&& keyframes)
        : keyframes_(std::forward<Keyframes>(keyframes)) {
        Initialize();
    }

    /// \param poses            (out) the new keyframe poses calculated as result of the optimization process
    /// \param inout            if true, utilize poses and locations both as input and output value. if false, poses and
    ///                         locations get initialized using the values in the underlying graph
    /// \param relative_motion  estimated relative motion between first and last keyframe
    /// \param parameters       parameters specifying the optimization process
    ///
    /// \returns            an overall residual value in case the optimization process was successful
    Result<double> Optimize(Poses& poses, bool inout, SE3d relative_motion, const Parameters& parameters);

private:
    void Initialize();

    /// Calculate the Jacobian of the problem instance
    ///
    /// The rows of the Jacobian correspond to the relative transformations of consecutive
    /// keyframes.
    ///
    /// That is, the rows are blocks of six rows each, in the order of the provided keyframes. The six rows correspond
    /// to the 6 dimensions of the logarithm of T_ij^-1 * T_j * T_i^-1, where j = (i + 1) mod N, N the number of
    /// keyframes.
    ///
    /// The columns correspond to the vector variables that we are optimizing: For each keyframe, we have 6 coordinates
    /// corresponding via exponential map to the pose in SE(3) associated with the second camera. Per documentation of
    /// ``Sophus::SE3::exp()``, the first three components of those coordinates represent the translational part
    /// ``upsilon`` in the tangent space of SE(3), while the last three components of ``a`` represents the rotation
    /// vector ``omega``.
    Eigen::SparseMatrix<double> CalculateJacobian(const SE3d& relative_motion, const Eigen::VectorXd& value) const;

    Eigen::VectorXd CalculateResidual(const SE3d& relative_motion, const Eigen::VectorXd& value) const;

    inline void 
    LoopPoseOptimizer::CalculateResidual0(Eigen::VectorXd& residual, const SE3d& relative_motion, 
                                          const Eigen::VectorXd& value, size_t from_index, size_t to_index, 
                                          size_t residual_index) const;

    enum {
        /// dimensionality of a pose (6)
        kDimPose = SE3d::DoF,

        /// dimensionality of a constraint
        kDimConstraint = SE3d::DoF
    };

    /// Return the offset associated with a given keyframe pose identified by its index within the
    /// overall vector of variables to optimize
    ///
    /// \param index    the index of the keyframe within the vector of keyframes
    /// \returns        the slot of values witjin the state variable vector
    inline auto PoseSlot(size_t index) const ->
        decltype(Eigen::seqN(index, int(kDimPose))) {
        assert(index < keyframes_.size());
        size_t start = index * kDimPose;
        return Eigen::seqN(start, int(kDimPose));
    }

    /// Return the overall dimension of the optimization problem
    size_t total_dimension() const { return keyframes_.size() * kDimPose; }

    /// Return the overall number of constraints of the optimization problem
    inline size_t total_constraints() const { return keyframes_.size() * kDimConstraint; }


    Keyframes keyframes_;
    std::vector<SE3d> relative_motion_;
};

} // namespace slammer

#endif //ndef SLAMMER_LOOP_POSE_OPTIMIZER_H
