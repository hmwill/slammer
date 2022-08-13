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

#ifndef SLAMMER_IMAGE_H
#define SLAMMER_IMAGE_H

#pragma once

#include "slammer/slammer.h"

#include <boost/gil.hpp>

namespace slammer {

/// Create an image pyramid for the provided image.
///
/// \param level_0  the original image at the root of the pyramid. 
///                 It is moved into the element at index 0 of the result.
/// \param scale    the scale factor between two adjacent images in the pyramid. The value
///                 should be in the range 0 < scale < 1.
/// \param num_levels the number of levels in the pyramid to generate
std::vector<boost::gil::gray8_image_t> 
CreatePyramid(boost::gil::gray8_image_t&& level_0, float scale, unsigned num_levels);

/// Create an image pyramid for the provided image.
///
/// \param level_0  the original image at the root of the pyramid. 
///                 It is moved into the element at index 0 of the result.
/// \param scale    the scale factor between two adjacent images in the pyramid. The value
///                 should be in the range 0 < scale < 1.
/// \param num_levels the number of levels in the pyramid to generate 
/// \param pyramid  a vector where the generated pyramid levels should be appended to
void AppendPyramidLevels(const boost::gil::gray8c_view_t& level_0, float scale, unsigned num_levels,
                         std::vector<boost::gil::gray8_image_t>& pyramid);

/// Convert an RGB image to a grayscale image
///
/// Note: This function is based on an example in the Boost GIL source tree and subject 
/// to the BOOST 1.0 license
/// 
/// Copyright 2019 Olzhas Zhumabek <anonymous.from.applecity@gmail.com>
///
/// \param original the image to convert
///
/// \return a grayscale image with same dimensions and resolution as the original
boost::gil::gray8_image_t RgbToGrayscale(const boost::gil::rgb8c_view_t& original);

/// Interface to allow for logging of images that are computed during the course of a
/// computation
class ImageLogger {
public:
    /// Log a grayscale image
    virtual void LogImage(const boost::gil::gray8c_view_t image, 
                          const std::string& name) = 0;

    /// Log a color image
    virtual void LogImage(const boost::gil::rgb8c_view_t image, 
                          const std::string& name) = 0;
};

class FileImageLogger: public ImageLogger {
public:
    FileImageLogger(const std::filesystem::path& prefix = std::filesystem::path()): prefix_(prefix) {}

    virtual void LogImage(const boost::gil::gray8c_view_t image, 
                          const std::string& name) override;

    virtual void LogImage(const boost::gil::rgb8c_view_t image, 
                          const std::string& name) override;

private:
    std::filesystem::path GetPath(const std::string& name, const std::string& suffix) const;

    std::filesystem::path prefix_;
};

} // namespace slammer

#endif //ndef SLAMMER_IMAGE_H
