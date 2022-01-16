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

#ifndef SLAMMER_ORB_H
#define SLAMMER_ORB_H

#pragma once

#include "slammer/slammer.h"

#include <boost/gil.hpp>

namespace slammer {

/// This namespace provides an implementation of ORB feature detection and matching.
/// It is written against the Boost Generic Image Library (https://github.com/boostorg/gil).
/// Initially, we are formulating the algorithms in a rather concrete form, but may
/// consider turning them into more generic versions lateron.
namespace orb {

/// Descriptor of an ORB feature
///
/// Overall, this structure utilizes 384 bits (48 bytes)
struct Descriptor {
    /// The decriptor is calculated using pixels within a circle with the following radius
    static constexpr int kRadius = 13;

    /// Number of sample pairs used to calculate the descriptor. This corresponds to the number
    /// of descriptor bits.
    static constexpr size_t kNumSamples = 256;

    using Bits = std::bitset<kNumSamples>;

    /// the image coordinate
    Point2f coords;

    /// the angle of the feature
    float angle;

    /// the level within the image pyramid
    uint32_t level;

    /// the bit pattern identiyfing the feature
    Bits descriptor_;
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