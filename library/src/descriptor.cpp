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

#include "slammer/descriptor.h"

using namespace slammer;

//
// Descriptor
//

Descriptor Descriptor::ComputeCentroid(const std::vector<const Descriptor*> descriptors) {
    std::array<unsigned, kNumBits> counters;
    counters.fill(0);

    for (auto descriptor: descriptors) {
        for (size_t index = 0; index < kNumBits; index += 8) {
            counters[index + 0] += descriptor->descriptor_[index + 0];
            counters[index + 1] += descriptor->descriptor_[index + 1];
            counters[index + 2] += descriptor->descriptor_[index + 2];
            counters[index + 3] += descriptor->descriptor_[index + 3];
            counters[index + 4] += descriptor->descriptor_[index + 4];
            counters[index + 5] += descriptor->descriptor_[index + 5];
            counters[index + 6] += descriptor->descriptor_[index + 6];
            counters[index + 7] += descriptor->descriptor_[index + 7];
        }
    }

    unsigned threshold = descriptors.size() / 2 + (descriptors.size() % 2);
    Descriptor result;

    for (size_t index = 0; index < kNumBits; ++index) {
        result.descriptor_[index] = counters[index] > threshold;
    }

    return result;
}

Descriptor::Distance Descriptor::ComputeDistance(const Descriptor& first, const Descriptor& second) {
    return (first.descriptor_ ^ second.descriptor_).count();
}

Descriptor::Descriptor(const uchar* bits) {
    for (auto block_count = kNumBits / sizeof(Bitset::block_type); block_count; --block_count) {
        Bitset::block_type block = 0;

        for (auto byte_count = sizeof(Bitset::block_type) / 8; byte_count; --byte_count) {
            block = (block << 8) | *bits++;
        }

        descriptor_.append(block);
    }
}

//
// DescriptorTree
//

