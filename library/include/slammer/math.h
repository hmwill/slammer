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

#ifndef SLAMMER_MATH_H
#define SLAMMER_MATH_H

#pragma once

#include "slammer/slammer.h"
#include "slammer/camera.h"

#include <random>

namespace slammer {

/// Determine an affine transformation that maps the `reference` point set onto the 
/// `transformed` poiint set, where corresponding points are stored at the same index.
///
/// This function calculates ICP using singular value decomposition (SVD).
///
/// \param reference    point coordinates in the reference frame
/// \param transformed  point coordinates in the transformed frame to be estimated
///
/// \return Estimated affine transformation
SE3d CalculateIcp(const std::vector<Point3d>& reference, const std::vector<Point3d>& transformed);

/// Determine an affine transformation that maps the `reference` point set onto the 
/// `transformed` poiint set, where corresponding points are stored at the same index.
///
/// This function calculates ICP using RANSAC based on singular value decomposition (SVD).
///
/// \param reference            point coordinates in the reference frame
/// \param transformed          point coordinates in the transformed frame to be estimated
/// \param[inout] random_engine the random engine to use for sample generation
/// \param[out] transformation  the estimated transformation
/// \param[out] inlier_mask     boolean vectors indicating the inliers
/// \param max_iterations       maximum number of iterations
/// \param sample_size          number of points to use for each estimate; if the total number of 
///                             points is less than this value, the function falls back to `CalculateIcp`.
/// \param threshold            Threshold value for outlier check
size_t RobustIcp(const std::vector<Point3d>& reference, const std::vector<Point3d>& transformed,
                 std::default_random_engine& random_engine,
                 SE3d& transformation, std::vector<uchar>& inlier_mask,
                 size_t max_iterations = 30, size_t sample_size = 10, double threshold = 7.16,
                 size_t min_additional_inliers = 20);

/// Determine an affine transformation that maps the `reference` point set onto the 
/// `transformed` point set, where corresponding points are stored at the same index.
///
/// This function calculates ICP using least squares optimization.
///
/// \param reference    point coordinates in the reference frame
/// \param transformed  point coordinates in the transformed frame to be estimated
/// \param mask         mask to exclude points from being considered
/// \param camera       description of the camera projection
/// \param baseline     distance between the two stereo cameras used for depth calculation via disparity
/// \param iterations   maximum number of iterations
///
/// \return Estimated affine transformation
SE3d OptimizeAlignment(const std::vector<Point3d>& reference, const std::vector<Point3d>& transformed,
                       const std::vector<uchar>& mask, const Camera& camera, double baseline, size_t iterations);

} // namespace slammer

#endif //ndef SLAMMER_MATH_H
