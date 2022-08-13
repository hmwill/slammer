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

#include <stdexcept>

#include <gtest/gtest.h>

#include "slammer/utility.h"

using namespace slammer;

TEST(UtilityTest, RotateLeft) {
    {
        uint8_t value = UINT8_C(0b11001010);
        EXPECT_EQ(RotateLeft(value, 1), UINT8_C(0b10010101));
        EXPECT_EQ(RotateLeft(value, 4), UINT8_C(0b10101100));
        EXPECT_EQ(RotateLeft(value, 7), UINT8_C(0b01100101));
    }

    {
        uint16_t value = UINT16_C(0b1100101010110010);
        EXPECT_EQ(RotateLeft(value,  1), UINT16_C(0b1001010101100101));
        EXPECT_EQ(RotateLeft(value,  8), UINT16_C(0b1011001011001010));
        EXPECT_EQ(RotateLeft(value, 15), UINT16_C(0b0110010101011001));
    }

    {
        uint32_t value = UINT32_C(0b11001010101100101011001011001010);
        EXPECT_EQ(RotateLeft(value,  1), UINT32_C(0b10010101011001010110010110010101));
        EXPECT_EQ(RotateLeft(value, 16), UINT32_C(0b10110010110010101100101010110010));
        EXPECT_EQ(RotateLeft(value, 31), UINT32_C(0b01100101010110010101100101100101));
    }

    {
        uint64_t value = UINT64_C(0b1100101010110010101100101100101010110010110010101100101010110010);
        EXPECT_EQ(RotateLeft(value, 1), UINT64_C(0b1001010101100101011001011001010101100101100101011001010101100101));
        EXPECT_EQ(RotateLeft(value, 32), UINT64_C(0b1011001011001010110010101011001011001010101100101011001011001010));
        EXPECT_EQ(RotateLeft(value, 63), UINT64_C(0b0110010101011001010110010110010101011001011001010110010101011001));
    }
}

TEST(UtilityTest, RotateRight) {
    {
        uint8_t value = UINT8_C(0b11001010);
        EXPECT_EQ(RotateRight(value, 1), UINT8_C(0b01100101));
        EXPECT_EQ(RotateRight(value, 4), UINT8_C(0b10101100));
        EXPECT_EQ(RotateRight(value, 7), UINT8_C(0b10010101));
    }

    {
        uint16_t value = UINT16_C(0b1100101010110010);
        EXPECT_EQ(RotateRight(value,  1), UINT16_C(0b0110010101011001));
        EXPECT_EQ(RotateRight(value,  8), UINT16_C(0b1011001011001010));
        EXPECT_EQ(RotateRight(value, 15), UINT16_C(0b1001010101100101));
    }

    {
        uint32_t value = UINT32_C(0b11001010101100101011001011001010);
        EXPECT_EQ(RotateRight(value,  1), UINT32_C(0b01100101010110010101100101100101));
        EXPECT_EQ(RotateRight(value, 16), UINT32_C(0b10110010110010101100101010110010));
        EXPECT_EQ(RotateRight(value, 31), UINT32_C(0b10010101011001010110010110010101));
    }

    {
        uint64_t value = UINT64_C(0b1100101010110010101100101100101010110010110010101100101010110010);
        EXPECT_EQ(RotateRight(value,  1), UINT64_C(0b0110010101011001010110010110010101011001011001010110010101011001));
        EXPECT_EQ(RotateRight(value, 32), UINT64_C(0b1011001011001010110010101011001011001010101100101011001011001010));
        EXPECT_EQ(RotateRight(value, 63), UINT64_C(0b1001010101100101011001011001010101100101100101011001010101100101));
    }
}

