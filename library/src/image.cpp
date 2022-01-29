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

#include "slammer/image.h"

#include "boost/gil/io/write_view.hpp"
#include "boost/gil/extension/io/png.hpp"

#include "boost/gil/extension/numeric/sampler.hpp"
#include "boost/gil/extension/numeric/resample.hpp"

using namespace slammer;
using namespace boost::gil;

std::vector<gray8_image_t> 
slammer::CreatePyramid(gray8_image_t&& level_0, float scale, unsigned num_levels) {
    std::vector<gray8_image_t> result;
    result.emplace_back(std::move(level_0));
    AppendPyramidLevels(const_view(result[0]), scale, num_levels, result);
    return result;
}

void 
slammer::AppendPyramidLevels(const boost::gil::gray8c_view_t& level_0, float scale, unsigned num_levels,
                             std::vector<boost::gil::gray8_image_t>& pyramid) {

    gray8c_view_t image = level_0;

    for (unsigned level = 0; level < num_levels; ++level) {
        auto dimensions = point_t(image.width() * scale, image.height() * scale);
        gray8_image_t new_image(dimensions);
        boost::gil::resize_view(image, view(new_image), boost::gil::bilinear_sampler{});
        pyramid.emplace_back(std::move(new_image));
        image = const_view(pyramid.back());
    }
}

boost::gil::gray8_image_t 
slammer::RgbToGrayscale(const boost::gil::rgb8c_view_t& original) {
    boost::gil::gray8_image_t output_image(original.dimensions());
    auto output = boost::gil::view(output_image);

    boost::gil::transform_pixels(original, output, [&](const auto& pixel) {
        constexpr float max_channel_intensity = std::numeric_limits<std::uint8_t>::max();
        constexpr float inv_max_channel_intensity = 1.0f / max_channel_intensity;

        const std::integral_constant<int, 0> kRed;
        const std::integral_constant<int, 1> kGreen;
        const std::integral_constant<int, 2> kBlue;

        // scale the values into range [0, 1] and calculate linear intensity
        auto linear_luminosity =
            (0.2126f * inv_max_channel_intensity) * pixel.at(kRed) + 
            (0.7152f * inv_max_channel_intensity) * pixel.at(kGreen) + 
            (0.0722f * inv_max_channel_intensity) * pixel.at(kBlue);

        // perform gamma adjustment
        float gamma_compressed_luminosity = 0;

        if (linear_luminosity < 0.0031308f) {
            gamma_compressed_luminosity = linear_luminosity * 12.92f;
        } else {
            gamma_compressed_luminosity = 1.055f * std::powf(linear_luminosity, 1 / 2.4f) - 0.055f;
        }

        // since now it is scaled, descale it back
        return gamma_compressed_luminosity * max_channel_intensity;
    });

    return output_image;
}

void 
FileImageLogger::LogImage(const boost::gil::gray8c_view_t image, 
                          const std::string& name) {
    auto path = GetPath(name, "png");
    boost::gil::write_view(path.string(), image, boost::gil::png_tag{});
}

void 
FileImageLogger::LogImage(const boost::gil::rgb8c_view_t image, 
                          const std::string& name) {
    auto path = GetPath(name, "png");
    boost::gil::write_view(path.string(), image, boost::gil::png_tag{});
}

std::filesystem::path 
FileImageLogger::GetPath(const std::string& name, const std::string& suffix) const {
    if (!std::filesystem::exists(prefix_)) {
        // TODO: Not sure yet about how to handle error situations here. For now, we just
        // eat the error silently, but will end up failing when the actual write operation is
        // called.
        std::error_code error;
        bool did_create = std::filesystem::create_directories(prefix_, error);
    }

    auto result = prefix_ / name;
    result.replace_extension(suffix);
    return result;
}