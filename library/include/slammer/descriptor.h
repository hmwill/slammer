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

#ifndef SLAMMER_DESCRIPTOR_H
#define SLAMMER_DESCRIPTOR_H

#pragma once

#include "slammer/slammer.h"

namespace slammer {

/// Representation of a binary feature descriptor; here, really, rotated BRIEF,
/// but the computation of the descriptor is hiddem within the feature detector.
///
/// Descriptors are fixed length bit strings and employ the Hamming distance as
/// natural pairwise distance function.
/// Description of a single image feature
///
/// For ORB features, which we are using, the feature descriptor is a 256-bit vector.
class Descriptor {
public:
    /// Number of bits per feature descriptor
    static const size_t kNumBits = 256;

    using Distance = size_t;
    using Collection = std::vector<Descriptor>;

    // We are using a dynamic bitset from the boost library until we have descriptor extraction
    // converted from not using OpenCV Mat types anymore.
    using Bitset = boost::dynamic_bitset<uint64_t>;

    Descriptor(): descriptor_(kNumBits) {}

    /// Set a specified descriptor bit
    ///
    /// \param index the index of the descriptor bit
    /// \param value the new bit value to set
    void Set(unsigned index, bool value = true) {
        descriptor_[index] = value;
    }

    /// Get a specified descriptor bit
    ///
    /// \param index the index of the descriptor bit
    bool Get(unsigned index) const {
        return descriptor_[index];
    }

    /// Ceate a feature descriptor based on the contents of a row in the given OpenCV matrix.
    /// The Matrix is expected to have elements of type CV_8UC1 and to have 32 columns.
    static Descriptor From(const cv::Mat& mat, int row)  {
        return Descriptor(mat.ptr(row));
    }   

    /// Ceate a feature descriptor based on the contents of all the rows in the given OpenCV matrix.
    /// The Matrix is expected to have elements of type CV_8UC1 and to have 32 columns.
    static Collection From(const cv::Mat& mat) {
        Collection result;

        for (int index = 0; index < mat.rows; ++index) {
            result.emplace_back(From(mat, index));
        }

        return result;
    }

    static void IntoPointerVector(const Collection& descriptors,
                                  std::vector<const Descriptor*>& result) {
        for (const auto& descriptor: descriptors) {
            result.push_back(&descriptor);
        }
    }

    /// Calculate the centroid of a collection of descriptors
    static Descriptor ComputeCentroid(const std::vector<const Descriptor*> descriptors);

    // Calculate the Hamming distance between two FeatureDescriptors
    static Distance ComputeDistance(const Descriptor& first, const Descriptor& second);

private:
    Descriptor(const uchar* bits);
    Descriptor(Bitset&& descriptor): descriptor_(descriptor) {}

    Bitset descriptor_;
};

/// A tree data structure to support matching of descriptors. The structure of the tree
/// can be learned using a recursive k-means clustering method. The tree can also be
/// serialized and recreated to support peristing the tree structure across program
/// executions.
class DescriptorTree {
public:

private:
};

} // namespace slammer

#endif //ndef SLAMMER_DESCRIPTOR_H
