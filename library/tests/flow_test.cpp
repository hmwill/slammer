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

#include "slammer/flow.h"

#include "boost/gil/extension/io/png.hpp"

using namespace slammer;

namespace {

using namespace boost::gil;

void LogFeatures(ImageLogger& logger, const std::string& name, const gray8c_view_t& image,
                 const std::vector<Point2f>& points, const std::vector<Point2f>* secondary = nullptr) {
    rgb8_image_t display_features(image.dimensions());
    auto output = view(display_features);

    copy_pixels(color_converted_view<rgb8_pixel_t>(image), output);

    const rgb8_pixel_t red(255, 0, 0);
    const rgb8_pixel_t blue(0, 0, 255);

    if (secondary) {
        for (auto point: *secondary) {
            output(point.x, point.y) = blue;
            output(point.x - 3, point.y) = blue;
            output(point.x - 2, point.y) = blue;
            output(point.x - 1, point.y) = blue;
            output(point.x + 1, point.y) = blue;
            output(point.x + 2, point.y) = blue;
            output(point.x + 3, point.y) = blue;
            output(point.x, point.y - 3) = blue;
            output(point.x, point.y - 2) = blue;
            output(point.x, point.y - 1) = blue;
            output(point.x, point.y + 1) = blue;
            output(point.x, point.y + 2) = blue;
            output(point.x, point.y + 3) = blue;
        }
    }

    for (auto point: points) {
        output(point.x, point.y) = red;
        output(point.x - 3, point.y) = red;
        output(point.x - 2, point.y) = red;
        output(point.x - 1, point.y) = red;
        output(point.x + 1, point.y) = red;
        output(point.x + 2, point.y) = red;
        output(point.x + 3, point.y) = red;
        output(point.x, point.y - 3) = red;
        output(point.x, point.y - 2) = red;
        output(point.x, point.y - 1) = red;
        output(point.x, point.y + 1) = red;
        output(point.x, point.y + 2) = red;
        output(point.x, point.y + 3) = red;
    }

    logger.LogImage(const_view(display_features), name);
}

}
// namespace

TEST(FlowTest, SimpleTest) {
    using namespace boost::gil;

    std::string kSourceInputPath("data/cafe1-1/color/1560004885.446172.png");
    //std::string kTargetInputPath("data/cafe1-1/color/1560004886.446856.png");
    std::string kTargetInputPath("data/cafe1-1/color/1560004887.013914.png");

    rgb8_image_t source_input, target_input;
    read_image(kSourceInputPath, source_input, png_tag{});
    read_image(kTargetInputPath, target_input, png_tag{});

    // make sure we have properly read those images
    ASSERT_EQ(source_input.dimensions(), target_input.dimensions());
    ASSERT_EQ(source_input.width(), 848);
    ASSERT_EQ(source_input.height(), 480);

    float sigma = 1.0f;
    size_t window = static_cast<size_t>(roundf(sigma * 3)) * 2 + 1;
    auto smoothing_kernel = generate_gaussian_kernel(window, sigma);

    auto source_gray = RgbToGrayscale(const_view(source_input));
    gray8_image_t source(source_gray.dimensions());
    detail::convolve_2d(const_view(source_gray), smoothing_kernel, view(source));

    auto target_gray = RgbToGrayscale(const_view(target_input));
    gray8_image_t target(target_gray.dimensions());
    detail::convolve_2d(const_view(target_gray), smoothing_kernel, view(target));

    std::vector<Point2f> source_points = {
        Point2f(372, 232),         Point2f(368, 240),
        Point2f(352, 280),         Point2f(311.127, 234.759),
        Point2f(420, 228),         Point2f(285.671, 322.441),
        Point2f(336.583, 285.671), Point2f(620, 208),
        Point2f(300, 236),         Point2f(336.583, 316.784),
        Point2f(704, 324),         Point2f(356, 292),
        Point2f(404, 148),         Point2f(610, 234),
        Point2f(652, 104),         Point2f(418.607, 237.588),
        Point2f(608.112, 76.3675), Point2f(352, 272),
        Point2f(316, 244),         Point2f(370, 300),
        Point2f(292, 276),         Point2f(358, 300),
        Point2f(328, 168),         Point2f(684, 112),
        Point2f(352, 336),         Point2f(720, 344),
        Point2f(622.254, 79.196),  Point2f(345.068, 254.558),
        Point2f(707.107, 330.926), Point2f(380, 132),
        Point2f(356, 400),         Point2f(296, 316),
        Point2f(333.754, 200.818), Point2f(328.098, 175.362),
        Point2f(708, 156),         Point2f(325.269, 285.671),
        Point2f(256, 248),         Point2f(695.793, 104.652),
        Point2f(792, 248),         Point2f(299.813, 299.813),
        Point2f(364, 250),         Point2f(408, 268),
        Point2f(200, 232),         Point2f(257.387, 313.955),
        Point2f(344, 244),         Point2f(676, 284),
        Point2f(421.436, 253.144), Point2f(452.548, 322.441),
        Point2f(296, 256),         Point2f(348, 224)
    };

    std::vector<Point2f> target_points;
    std::vector<float> error;

    ComputeFlow(const_view(source), const_view(target), source_points, target_points, error);
    EXPECT_EQ(target_points.size(), source_points.size());

    for (auto err: error) {
        EXPECT_LE(sqrt(err), 0.5);
    }

    if (true) {
        FileImageLogger logger("image_logs/flow_test/simple_test");
        LogFeatures(logger, "Input", const_view(source_gray), source_points);
        LogFeatures(logger, "Output", const_view(target_gray), target_points, &source_points);
    }
}