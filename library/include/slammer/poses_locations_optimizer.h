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

#ifndef SLAMMER_POSES_LOCATIONS_OPTIMIZER_H
#define SLAMMER_POSES_LOCATIONS_OPTIMIZER_H

#pragma once

#include "slammer/slammer.h"
#include "slammer/camera.h"
#include "slammer/map.h"

#include "Eigen/Sparse"

namespace slammer {

/// Instance of a pose graph optimization problem.
///
/// We are given a bipartite graph of keyframes and landmarks alongside a current
/// estimates of keyframe poses and landmark locations. For each landmark, the
/// graph structure indicates if it has been obsevred in a given keyframe, and if
/// so, what the projected image location and detected depth value are. The (x,y)
/// coordinates are the 2D image location within a standard (RGB) camera image, and
/// the z coordinate is the depth sensor value. For the model here, we assume that 
/// the depth value has been derived by determining the dispersion across a pair of
/// stereo images, as is, for example, the case for an Intel Realsense camera.
class PosesLocationsOptimizer {
public:
    struct Parameters {
        /// the numerical optimization algorithm to employ
        enum Algorithm {
            kGaussNewton,
            kLevenbergMarquardt
        } algorithm;

        /// maximum number of iterations
        unsigned max_iterations;

        /// the lambda parameter for Levenberg-Marquardt
        double lambda;
    };

    using Keyframes = std::vector<KeyframePointer>;
    using Landmarks = std::vector<LandmarkPointer>;
    using Poses = std::vector<SE3d>;
    using Locations = std::vector<Point3d>;

    using LandmarkMapping = std::unordered_map<LandmarkId, LandmarkId>;

    /// Construct and instance of the pose graph optimization problem
    ///
    /// \param depth_camera the camera utilized to capture the keyframes, which determines the error function
    /// \param keyframes    the keyframes included in the subgraph that are subject to optimization
    /// \param landmarks    the landmarks included in the subgraph that are subject to optimization
    /// \param mapping      assume the unification of landmarks based on this mapping    
    template<class C, class K, class L, class M>
    PosesLocationsOptimizer(C&& depth_camera, K&& keyframes, L&& landmarks, M&& mapping)
        : depth_camera_(std::forward<StereoDepthCamera>(depth_camera)), 
          keyframes_(std::forward<Keyframes>(keyframes)), 
          landmarks_(std::forward<Landmarks>(landmarks)), 
          mapping_(std::forward<LandmarkMapping>(mapping)) {
        total_dimension_ = keyframes_.size() * kDimPose + landmarks_.size() * kDimLocation;

        // create reverse lookup table for keyframes
        for (size_t index = 0; index < keyframes_.size(); ++index) {
            assert(keyframe_index_.find(keyframes_[index]) == keyframe_index_.end());
            keyframe_index_.insert({ keyframes_[index], index });
        }

        // create reverse lookup table for landmarks
        for (size_t index = 0; index < landmarks_.size(); ++index) {
            assert(landmark_index_.find(landsmarks_[index]) == landmark_index_.end());
            landmark_index_.insert({ landmarks_[index], index });
        }

        CalculateConstraints();
    }

    /// \param poses        (out) the new keyframe poses calculated as result of the optimization process
    /// \param locations    (out) the new landmark locations calculated as result of the optimization process
    /// \param inout        if true, utilize poses and locations both as input and output value. if false, poses and
    ///                     locations get initialized using the values in the underlying graph
    /// \param parameters   parameters specifying the optimization process
    ///
    /// \returns            an overall residual value in case the optimization process was successful
    Result<double> Optimize(Poses& poses, Locations& locations, bool inout, const Parameters& parameters);

protected:
    /// Calculate the Jacobian of the problem instance
    /// The rows of the Jacobian correspond to the measurement values associated with the 
    /// point pairs.
    ///
    /// That is, the rows are blocks of six rows each, in the order of the provided point
    /// pairs. Each block has two parts corresponding to the 3 coordinates of the first
    /// measurement followed by the 3 coordinates of the second measurement.
    ///
    /// The columns correspond to the vector variables that we are optimizing. The initial
    /// 6 define the tangent corresponding via exponential map to the pose in SE(3)
    /// associated with the second camera. Per documentation of ``Sophus::SE3::exp()``,
    /// the first three components of those coordinates represent the translational part
    /// ``upsilon`` in the tangent space of SE(3), while the last three components
    /// of ``a`` represents the rotation vector ``omega``.
    ///
    /// Then, for each point pair, we have an entry corresponding to (x, y, z) coordinates 
    /// of the meassured feature point in 3D world space.
    Eigen::SparseMatrix<double> CalculateJacobian(const Eigen::VectorXd& value) const;

    Eigen::VectorXd CalculateResidual(const Eigen::VectorXd& value) const;

private:
    enum {
        /// locations are 3-dimensional
        kDimLocation = 3,

        /// dimensionality of a pose (6)
        kDimPose = SE3d::DoF,

        /// dimensionality of a constraint
        kDimConstraint = 3
    };

    /// Return the offset associated with a given location identified by its index within the
    /// overall vector of variables to optimize
    ///
    /// \param index    the index of the location within the vector of landmarks
    /// \returns        the slot of values within the state variable vector
    inline auto LocationSlot(size_t index) const -> 
        decltype(Eigen::seqN(index, int(kDimLocation))) {
        assert(index < landmarks_.size());
        return Eigen::seqN(index * kDimLocation, int(kDimLocation));
    }

    /// Return the offset associated with a given keyframe pose identified by its index within the
    /// overall vector of variables to optimize
    ///
    /// \param index    the index of the keyframe within the vector of keyframes
    /// \returns        the slot of values witjin the state variable vector
    inline auto PoseSlot(size_t index) const ->
        decltype(Eigen::seqN(index, int(kDimPose))) {
        assert(index < keyframes_.size());
        size_t start = index * kDimPose + landmarks_.size() * kDimLocation;
        return Eigen::seqN(start, int(kDimPose));
    }

    /// Return the overall dimension of the optimization problem
    inline size_t total_dimension() const {
        return total_dimension_;
    }

    /// Return the overall number of constraints of the optimization problem
    inline size_t total_constraints() const {
        return total_constraints_;
    }

    void CalculateConstraints();

    StereoDepthCamera depth_camera_;
    Keyframes keyframes_;
    Landmarks landmarks_;
    LandmarkMapping mapping_;

    std::unordered_map<KeyframePointer, size_t> keyframe_index_;
    std::unordered_map<LandmarkPointer, size_t> landmark_index_;

    size_t total_dimension_;
    size_t total_constraints_;
};


} // slammer

#endif //ndef SLAMMER_POSES_LOCATIONS_OPTIMIZER_H
