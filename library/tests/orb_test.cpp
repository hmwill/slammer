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

#include "slammer/orb.h"

#include "boost/gil/extension/io/png.hpp"


using namespace slammer;
using namespace slammer::orb;


TEST(OrbTest, Basic) {
    Parameters parameters;
    Detector detector(parameters);
    FileImageLogger logger("image_logs/orb_test/basic");

    std::string kInputPath("data/cafe1-1/color/1560004885.446172.png");
    boost::gil::rgb8_image_t input;
    boost::gil::read_image(kInputPath, input, boost::gil::png_tag{});

    std::vector<KeyPoint> features;
    Descriptors descriptors;
    size_t num_features = detector.ComputeFeatures(const_view(input), 50, features, &descriptors, nullptr, &logger);
    EXPECT_EQ(features.size(), num_features);
    EXPECT_LE(features.size(), 50);

    for (const auto& feature: features) {
        std::cout << "(" << feature.coords.x << ", " << feature.coords.y << ")" << std::endl;
    }
} 