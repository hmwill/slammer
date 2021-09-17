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

#include <stdexcept>

#include <gtest/gtest.h>

#include "slammer/slammer.h"

using namespace slammer;

TEST(SlammerTests, TestError) {
    Error err1("This is a message");
    EXPECT_EQ(err1.message(), std::string("This is a message"));

    try {
        throw std::logic_error("This is the what");
    }
    catch (...) {
        Error error = Error::FromCurrentException();
        EXPECT_EQ(error.message(), std::string("This is the what"));
    }
}

TEST(SlammerTests, TestResult) {
    Result<int> ok_result { 10 };
    Result<int> fail_result { std::string("This is an error") };

    EXPECT_TRUE(ok_result.is_ok());
    EXPECT_FALSE(ok_result.is_error());
    EXPECT_EQ(ok_result.value(), 10);

    EXPECT_FALSE(fail_result.is_ok());
    EXPECT_TRUE(fail_result.is_error());
    EXPECT_EQ(fail_result.error().message(), std::string("This is an error"));

    Result copy_ok(ok_result);
    EXPECT_EQ(copy_ok.value(), 10);

    Result move_ok(std::move(ok_result));
    EXPECT_EQ(move_ok.value(), 10);
    EXPECT_EQ(ok_result.value(), 0);
}