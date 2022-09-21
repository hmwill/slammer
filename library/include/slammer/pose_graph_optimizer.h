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

#ifndef SLAMMER_POSE_GRAPH_OPTIMIZER_H
#define SLAMMER_POSE_GRAPH_OPTIMIZER_H

#pragma once

#include "slammer/slammer.h"
#include "slammer/map.h"

#include "Eigen/Sparse"

// Forward declaration of a test case class
class PoseGraphOptimizer_Derivatives_Test;

namespace slammer {

/// Optimize poses using covisibility constraints and any additional constraints provided.
///
/// The algorithm is based on the optimization problem formulation in
/// Aloise, Irvin, and Giorgio Grisetti. 2018. “Matrix Difference in Pose-Graph Optimization.” 
/// arXiv [cs.RO]. arXiv. http://arxiv.org/abs/1809.00952.
class PoseGraphOptimizer {
public:
    /// Paramters for the optimization algorithm
    struct Parameters {
        /// maximum number of iterations
        unsigned max_iterations;

        /// the lambda parameter for Levenberg-Marquardt
        double lambda;
    };

    /// Pose graph constraint representation
    struct Constraint {
        /// Originating vertex in the pose graph
        Keyframe * from;

        /// Destination vertex in the pose graph
        Keyframe * to;

        /// Relative motion as we move from the originating vertex to the destination vertex
        SE3d relative_motion;
    };

    using Constraints = std::vector<Constraint>;
    using Keyframes = std::vector<Keyframe *>;
    using KeyframeIndex = std::unordered_map<const Keyframe *, size_t>;
    using Poses = std::vector<SE3d>;

    /// Construct an instance of the pose graph optimization problem
    ///
    /// \param keyframes    the keyframes included in the subgraph that are subject to optimization
    PoseGraphOptimizer(const Keyframes& keyframes)
        : keyframes_(keyframes) { 
        index_ = CreateIndex(keyframes_);
        num_graph_constraints_ = CalculateGraphConstraints(keyframes_, index_);    
    }

    /// Construct an instance of the pose graph optimization problem
    ///
    /// \param keyframes    the keyframes included in the subgraph that are subject to optimization
    PoseGraphOptimizer(Keyframes&& keyframes)
        : keyframes_(std::forward<Keyframes>(keyframes)) { 
        index_ = CreateIndex(keyframes_);
        num_graph_constraints_ = CalculateGraphConstraints(keyframes_, index_);    
    }

    /// \param poses            (out) the new keyframe poses calculated as result of the optimization process
    /// \param inout            if true, utilize poses and locations both as input and output value. if false, poses and
    ///                         locations get initialized using the values in the underlying graph
    /// \param constraints      additional constraints to incorporate into optimization problem
    /// \param parameters       parameters specifying the optimization process
    ///
    /// \returns            an overall residual value in case the optimization process was successful
    Result<double> Optimize(Poses& poses, bool inout, const Constraints& constraints, const Parameters& parameters);

    /// Calculate the relative motion as we traverse an edge in the pose graph
    ///
    /// \param from             the pose graph vertex the edge is starting from
    /// \param to               the pose graph vertex the edge ends at
    ///
    /// \returns the reltive motion expressed as element in SE(3)
    inline static SE3d CalculateRelativeMotion(const Keyframe* from, const Keyframe* to) {
        return to->pose.inverse() * from->pose;
    }

private:
    enum {
        /// dimensionality of a pose (6)
        kDimPose = 6,

        /// dimensionality of a constraint, which is the 12 entries of the first 3 rows of the transformation matrix
        kDimConstraint = 12,
    };

    /// Create the reverse index mapping keyframe pointer to index within keyframes vector
    static inline KeyframeIndex CreateIndex(const Keyframes& keyframes) {
        KeyframeIndex result;

        for (size_t index = 0; index < keyframes.size(); ++index) {
            result[keyframes[index]] = index;
        }

        return result;
    }

    /// Calculate the number of constraints implied by the pose graph structure
    static inline size_t CalculateGraphConstraints(const Keyframes& keyframes, const KeyframeIndex& index) {
        size_t count = 0;

        for (auto keyframe: keyframes) {
            for (auto neighbor: keyframe->covisible) {
                auto neighbor_iter = index.find(neighbor.get());

                if (neighbor_iter == index.end()) {
                    // consraints beyond the keyframe poses to optimize are always counted
                    ++count;
                } else {
                    // ensure that we count constraints amongst the keyframe poses to optimize only once
                    count += neighbor->timestamp > keyframe->timestamp;
                }
            }
        }

        return count;
    }

    /// Calculate the Jacobian of the problem instance
    ///
    /// The rows of the Jacobian correspond to the relative transformations of consecutive keyframes.
    ///
    /// That is, the rows are blocks of 12 rows each, in the order of the provided keyframes. The 12 rows correspond
    /// to the 12 dimensions of the top 3 rows of T_ij^-1 * T_j * T_i^-1, where j = (i + 1) mod N, N the number of
    /// keyframes.
    ///
    /// The columns correspond to the vector variables that we are optimizing: For each keyframe, we have 6 coordinates
    /// corresponding to a 3-dimensional translation and angles around 3 axes of rotation.
    Eigen::SparseMatrix<double> CalculateJacobian(const Poses& poses, const Constraints& constraints, 
                                                  const Eigen::VectorXd& value) const;

    Eigen::VectorXd CalculateResidual(const Poses& poses, const Constraints& constraints, 
                                      const Eigen::VectorXd& value) const;

    inline void 
    PoseGraphOptimizer::CalculateResidual0(const Poses& poses, Eigen::VectorXd& residual, const SE3d& relative_motion, 
                                          const Eigen::VectorXd& value, const Keyframe* from, const Keyframe* to, 
                                          size_t residual_index) const;

    // Allow access to the following two static functions from test case
    friend class ::PoseGraphOptimizer_Derivatives_Test;

    using PoseParameters = Eigen::Vector<double, kDimPose>;

    /// Calculate an SE(3) element from the provided parameters
    static SE3d TransformFromParameters(const PoseParameters& params);

    /// Calculate the Jacobian for the parameters of a relative motion
    static Eigen::Matrix<double, kDimConstraint, kDimPose> 
    CalculateJacobianComponent(const SE3d& after, const PoseParameters& params, const SE3d& before);
    
    /// Return the offset associated with a given keyframe pose identified by its index within the
    /// overall vector of variables to optimize
    ///
    /// \param index    the index of the keyframe within the vector of keyframes; we do not have a slot for index 0
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
    inline size_t total_constraints(const Constraints& additional) const { 
        return (num_graph_constraints_ + additional.size()) * kDimConstraint; 
    }

    // the keyframes subject to optimization
    Keyframes keyframes_;

    // reverse look-up from pointer to index within keyframes vector
    KeyframeIndex index_;

    // number of constraints within existing pose graph
    size_t num_graph_constraints_;
};

} // namespace slammer

#endif //ndef SLAMMER_POSE_GRAPH_OPTIMIZER_H
