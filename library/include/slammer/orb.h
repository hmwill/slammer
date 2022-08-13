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

#ifndef SLAMMER_ORB_H
#define SLAMMER_ORB_H

#pragma once

#include "slammer/slammer.h"

#include "slammer/descriptor.h"
#include "slammer/image.h"

namespace slammer {

/// This namespace provides an implementation of ORB feature detection and matching.
/// It is written against the Boost Generic Image Library (https://github.com/boostorg/gil).
/// Initially, we are formulating the algorithms in a rather concrete form, but may
/// consider turning them into more generic versions lateron.
namespace orb {

/// Representation of an ORB key point
///
/// Overall, this structure utilizes 384 bits (48 bytes)
struct KeyPoint {
    /// the image coordinate
    Point2f coords;

    /// the angle of the feature
    float angle;

    /// the level within the image pyramid
    uint32_t level;
};

/// The parameters needed to fully specify ORB feature detection and descriptor
struct Parameters {
    /// Minimum distance between returned feature points
    int min_distance = 10;

    /// Number of levels in the image pyramid
    unsigned levels = 4;

    /// Scale factor between levels in the pyramid
    float scale_factor = sqrtf(0.5f);

    /// FAST detection threshold
    uint8_t threshold = 20;

    /// window size for calculating orientation
    int window_size = 7;
    
    /// Variance (sigma) of smoothing filter
    float sigma = 1.0;

    /// `k`-parameter in Harris score
    float k = 0.15;
};

/// The kernels to use for computing the Harris scores
struct HarrisKernels {
    /// The kernel to use for calculating the gradient in x direction 
    boost::gil::detail::kernel_2d<float> dx;

    /// The kernel to use for calculating the gradient in y direction
    boost::gil::detail::kernel_2d<float> dy;

    /// The smoothing kernel to use for aggregating across tensor values
    boost::gil::detail::kernel_2d<float> smoothing;
};

class Detector {
public:
    Detector(const Parameters& parameters);

    /// Detect ORB features and compute their descriptor values
    ///
    /// \param original     the original RGB image in which we want to detect feature points
    /// \param max_features the maximum number of features to return
    /// \param result       container to receive the collection of detected features
    /// \param descriptors  if not nullptr, receives the feature descriptors
    /// \param feature_mask an optional image map that serves as stencil for feature selection
    ///
    /// \returns the number of detected features
    size_t ComputeFeatures(const boost::gil::rgb8c_view_t& original, 
                           size_t max_features, 
                           std::vector<KeyPoint>& result,
                           Descriptors* descriptors = nullptr,
                           const boost::gil::gray8c_view_t* feature_mask = nullptr,
                           ImageLogger * logger = nullptr) const;

private:
    Parameters parameters_;
    HarrisKernels kernels_;
};

} // namespace orb
} // namespace slammer

#endif //ndef SLAMMER_ORB_H

/* 
TODO:

- Build image pyramid
- Detect corners
- Determine level and rotation
- Calculate descriptor bit masks
- Pairwise matching of features

*/