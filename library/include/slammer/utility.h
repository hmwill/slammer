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

#ifndef SLAMMER_UTILITY_H
#define SLAMMER_UTILITY_H

#pragma once

#include "slammer/slammer.h"

namespace slammer {

/// Left-rotation of a 16-bit bit pattern
template <typename UnsignedInt>
inline constexpr UnsignedInt RotateLeft(UnsignedInt value, unsigned shift) {
    static_assert(std::numeric_limits<UnsignedInt>::is_integer &&
                  !std::numeric_limits<UnsignedInt>::is_signed);

    constexpr unsigned kNumBits = 8 * sizeof(value);
    constexpr UnsignedInt kMaxValue = std::numeric_limits<UnsignedInt>::max();
    return ((value << shift) | (value >> (kNumBits - shift))) & kMaxValue;
}

/// Right-rotation of a 16-bit bit pattern
template <typename UnsignedInt>
inline constexpr UnsignedInt RotateRight(UnsignedInt value, unsigned shift) {
    static_assert(std::numeric_limits<UnsignedInt>::is_integer &&
                  !std::numeric_limits<UnsignedInt>::is_signed);

    constexpr unsigned kNumBits = 8 * sizeof(value);
    constexpr UnsignedInt kMaxValue = std::numeric_limits<UnsignedInt>::max();
    return ((value >> shift) | (value << (kNumBits - shift))) & kMaxValue;
}

} // namespace slammer

#endif //ndef SLAMMER_UTILITY_H
