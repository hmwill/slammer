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

#ifndef SLAMMER_PNP_H
#define SLAMMER_PNP_H

#pragma once

#include "slammer/slammer.h"
#include "slammer/camera.h"

#include "Eigen/Sparse"

namespace slammer {

/// Instance of a perspective and point problem for two sets of point generated with
/// an RGB-D camera.
///
/// We are given two sets of coordinates, where entries at the same index correspond
/// to measurements of the same feature in 3D from two camera locations. The (x,y)
/// coordinates are the 2D image location within a standard (RGB) camera image, and
/// the z coordinate is the depth sensor value. For the model here, we assume that 
/// the depth value has been derived by determining the dispersion across a pair of
/// stereo images, as is, for example, the case for an Intel Realsense camera.
/// We also assume that the first camera is placed at the origin and looking in 
/// (negative?) Z direction.
class PerspectiveAndPoint3d {
public:
    typedef std::pair<Point3d, Point3d> PointPair;
    typedef std::vector<PointPair> PointPairs;

    PerspectiveAndPoint3d(const StereoDepthCamera& first, const StereoDepthCamera& second,
                          const PointPairs& point_pairs)
        : first_(first), second_(second), point_pairs_(point_pairs) {}

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

    Result<double> SolveLevenberg(SE3d& pose, std::vector<Point3d>& points, unsigned max_iterations = 10, 
                                  double lambda = 0.1);

    Result<double> Ransac(SE3d& pose, std::vector<uchar>& inlier_mask, const std::vector<Point3d>& points, 
                          size_t sample_size, unsigned max_iterations, double lambda, double threshold,
                          std::default_random_engine& random_engine);

private:
    StereoDepthCamera first_;
    StereoDepthCamera second_;
    PointPairs point_pairs_;
};


} // slammer

#endif //ndef SLAMMER_PNP_H
