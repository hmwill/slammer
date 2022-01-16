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

#include "slammer/orb.h"

using namespace slammer;
using namespace slammer::orb;

using namespace boost::gil;

namespace {

/// Calculate the pixel values of the output image by downasmpling the input image.
///
/// \param in   the input imageto down sample
/// \param out  the output image to be computed
void DownSample(const gray8c_view_t& in, gray8_view_t& out) {
    if (in.dimensions() == out.dimensions()) {
        // TODO: Special case for same size, where we just copy the images
    }

    assert(out.width() <= in.width());
    assert(out.height() <= in.height());

    // TODO: Replace mixed floating point and integer computation by fixed precision,
    // such as 16-bit fractions per pixel
    float scale_x = static_cast<float>(in.width()) / static_cast<float>(out.width());
    float scale_y = static_cast<float>(in.height()) / static_cast<float>(out.height());

    float inv_area = 1.0 / scale_x * scale_y;

    int window_x = ceilf(scale_x);
    int window_y = ceilf(scale_y);

    for (gray8_view_t::y_coord_t y = 0; y < out.height(); ++y) {
        gray8c_view_t::y_coord_t y_min = floorf(y * scale_y);
        gray8c_view_t::y_coord_t y_max = y_min + window_y - 1;
        float factor_y_min = ceilf(y * scale_y) - y * scale_y;
        float factor_y_max = ceilf((y + 1) * scale_y) - (y + 1) * scale_y;

        for (gray8_view_t::x_coord_t x = 0; x < out.width(); ++x) {
            gray8c_view_t::y_coord_t x_min = floorf(x * scale_x);
            gray8c_view_t::y_coord_t x_max = x_min + window_x - 1;
            float factor_x_min = ceilf(x * scale_x) - x * scale_x;
            float factor_x_max = ceilf((x + 1) * scale_x) - (x + 1) * scale_x;

            float sum = 0.0f;

            sum += factor_x_min * factor_y_min * in(x_min, y_min);
            sum += factor_x_max * factor_y_min * in(x_max, y_min);
            sum += factor_x_min * factor_y_max * in(x_min, y_max);
            sum += factor_x_max * factor_y_max * in(x_max, y_max);

            for (gray8_view_t::x_coord_t index_x = x_min + 1; index_x < x_max; ++index_x) {
                sum += factor_y_min * in(index_x, y_min);
                sum += factor_y_max * in(index_x, y_max);
            }

            // TODO: Special case for scale is between 0.5 and 1.0, where we do not need this block
            for (gray8_view_t::y_coord_t index_y = y_min + 1; index_y < y_max; ++index_y) {
                for (gray8_view_t::x_coord_t index_x = x_min + 1; index_x < x_max; ++index_x) {
                    sum += in(index_x, index_y);
                }
            }

            out(x, y) = static_cast<gray8_view_t::value_type>(sum * inv_area);
        }
    }
}

/// Create an image pyramid for the provided image.
///
/// \param level_0  the original image at the root of the pyramid. 
///                 It is moved into the element at index 0 of the result.
/// \param scale    the scale factor between two adjacent images in the pyramid. The value
///                 should be in the range 0 < scale < 1.
/// \param num_levels the number of levels in the pyramid to generate
std::vector<gray8_image_t> CreatePyramid(gray8_image_t&& level_0, float scale, unsigned num_levels) {
    std::vector<gray8_image_t> result;

    result.emplace_back(std::move(level_0));

    for (unsigned level = 0; level < num_levels; ++level) {
        auto& image = result[level];
        auto width = image.width() * scale;
        auto height = image.height() * scale;
        auto dimensions = point_t(width, height);

        gray8_image_t new_image(dimensions);
        auto view_in = const_view(image);
        auto view_out = view(new_image);
        DownSample(view_in, view_out);

        result.emplace_back(std::move(new_image));
    }

    return result;
}

/// Left-rotation of a 16-bit bit pattern
inline uint16_t RotateLeft(uint16_t value, unsigned shift) {
    return ((value << shift) | (value >> (16 - shift))) & std::numeric_limits<uint16_t>::max();
}

/// Detect FAST features in the given image and return a collection of feature point coordinates.
///
/// \param image The image bitmap where we are looking for FAST features. This needs to be a
///                 single channel gray-scale image.
/// \param threshold    The intensity threshold to apply in the detection process
inline std::vector<Point2i> DetectFastFeatures(const gray8c_view_t& image, float threshold) {
    static constexpr uint16_t kLower12 = (1u << 12) - 1;

    std::vector<Point2i> result;

    if (image.width() <= Descriptor::kRadius || image.height() <= Descriptor::kRadius) {
        return result;
    }

    gray8c_view_t::xy_locator center = image.xy_at(Descriptor::kRadius, Descriptor::kRadius);
    gray8c_view_t::xy_locator::cached_location_t locations[16]  = {
        center.cache_location(0,-3),
        center.cache_location(1,-3),
        center.cache_location(2,-2),
        center.cache_location(3,-1),
        center.cache_location(3, 0),
        center.cache_location(3, 1),
        center.cache_location(2, 2),
        center.cache_location(1, 3),
        center.cache_location(0, 3),
        center.cache_location(-1,3),
        center.cache_location(-2,2),
        center.cache_location(-3,1),
        center.cache_location(-3,0),
        center.cache_location(-3,-1),
        center.cache_location(-2,-2),
        center.cache_location(-1,-3)
    };

    for (int y = Descriptor::kRadius; y < image.height() - 2 * Descriptor::kRadius - 1; ++y) {
        for (int x = Descriptor::kRadius; x < image.width() - 2 * Descriptor::kRadius - 1; ++x) {
            auto reference = *center;

            uint16_t above = 0, below = 0;
            constexpr auto kMaxValue = std::numeric_limits<decltype(reference)>::max();

            // TODO: Should run benchmark to see if it is worth implementing the simplified
            // elimination test involving 2/4 points at 180/90 degree angle
            if (reference >= threshold) {
                for (unsigned index = 0, mask = 1; index < 16; ++index, mask <<= 1) {
                    auto value = center[locations[index]];
                    below |= mask * (value <= reference - threshold);
                }
            }

            if (below & kLower12 == kLower12 ||
                RotateLeft(below, 1) & kLower12 == kLower12 ||
                RotateLeft(below, 2) & kLower12 == kLower12 ||
                RotateLeft(below, 3) & kLower12 == kLower12 ||
                RotateLeft(below, 4) & kLower12 == kLower12) {
                result.emplace_back(x, y);
            } else if (reference <= kMaxValue - threshold) {
                for (unsigned index = 0, mask = 1; index < 16; ++index, mask <<= 1) {
                    auto value = center[locations[index]];
                    above |= mask * (value >= reference + threshold);
                }

                if (above & kLower12 == kLower12 ||
                    RotateLeft(above, 1) & kLower12 == kLower12 ||
                    RotateLeft(above, 2) & kLower12 == kLower12 ||
                    RotateLeft(above, 3) & kLower12 == kLower12 ||
                    RotateLeft(above, 4) & kLower12 == kLower12) {
                    result.emplace_back(x, y);
                }
            }

            ++center.x();
        }

        center += point<std::ptrdiff_t>(2 * Descriptor::kRadius - image.width(), 1);
    }

    return result;
}

/// Calculate the orientation of a feature anchored at a given coordinate
///
/// \param image    the image bitmap
/// \param coords   the coordinates of the center of the feature point within the image
/// \param delta    half-window size for computation (windows is 2*delta+1)
///
/// \return the orientation of the feature expressed as angle in radians
inline float CalculateOrientation(const gray8c_view_t& image, Point2i coords, int delta) {
    assert(delta > 0);

    // obtain the feature center; we use floor as rounding mode because any
    // point within a pixel square should be mapped to the pixel itself
    ptrdiff_t integer_x = static_cast<ptrdiff_t>(floorf(coords.x));
    ptrdiff_t integer_y = static_cast<ptrdiff_t>(floorf(coords.y));

    gray8c_view_t::xy_locator center = image.xy_at(integer_x, integer_y);
    float m00 = 0.0f, m10 = 0.0f, m01 = 0.0f;

    for (int delta_y = -delta; delta_y <= delta; ++delta_y) {
        for (int delta_x = -delta; delta_x <= delta; ++delta_x) {
            auto sample = center(delta_x, delta_y);
            m10 += delta_x * sample;
            m01 += delta_y * sample;
        }
    }

    return atan2f(m01, m10);
}

/// Table of sample point coordinates relative to the image center for calculating
/// BRIEF descriptor values. This table here is the set of numbers used by OpenCV
/// in order to facilitate testing and validation. Once the library has been stabilized,
/// we may want to generate an independent set of points.
const struct {
    struct { int8_t x, y; } s0, s1;
} kSampleCoordinates[256] = {
    {{8, -3}, {9, 5}},      {{4, 2}, {7, -12}},      {{-11, 9}, {-8, 2}},
    {{7, -12}, {12, -13}},  {{2, -13}, {2, 12}},     {{1, -7}, {1, 6}},
    {{-2, -10}, {-2, -4}},  {{-13, -13}, {-11, -8}}, {{-13, -3}, {-12, -9}},
    {{10, 4}, {11, 9}},     {{-13, -8}, {-8, -9}},   {{-11, 7}, {-9, 12}},
    {{7, 7}, {12, 6}},      {{-4, -5}, {-3, 0}},     {{-13, 2}, {-12, -3}},
    {{-9, 0}, {-7, 5}},     {{12, -6}, {12, -1}},    {{-3, 6}, {-2, 12}},
    {{-6, -13}, {-4, -8}},  {{11, -13}, {12, -8}},   {{4, 7}, {5, 1}},
    {{5, -3}, {10, -3}},    {{3, -7}, {6, 12}},      {{-8, -7}, {-6, -2}},
    {{-2, 11}, {-1, -10}},  {{-13, 12}, {-8, 10}},   {{-7, 3}, {-5, -3}},
    {{-4, 2}, {-3, 7}},     {{-10, -12}, {-6, 11}},  {{5, -12}, {6, -7}},
    {{5, -6}, {7, -1}},     {{1, 0}, {4, -5}},       {{9, 11}, {11, -13}},
    {{4, 7}, {4, 12}},      {{2, -1}, {4, 4}},       {{-4, -12}, {-2, 7}},
    {{-8, -5}, {-7, -10}},  {{4, 11}, {9, 12}},      {{0, -8}, {1, -13}},
    {{-13, -2}, {-8, 2}},   {{-3, -2}, {-2, 3}},     {{-6, 9}, {-4, -9}},
    {{8, 12}, {10, 7}},     {{0, 9}, {1, 3}},        {{7, -5}, {11, -10}},
    {{-13, -6}, {-11, 0}},  {{10, 7}, {12, 1}},      {{-6, -3}, {-6, 12}},
    {{10, -9}, {12, -4}},   {{-13, 8}, {-8, -12}},   {{-13, 0}, {-8, -4}},
    {{3, 3}, {7, 8}},       {{5, 7}, {10, -7}},      {{-1, 7}, {1, -12}},
    {{3, -10}, {5, 6}},     {{2, -4}, {3, -10}},     {{-13, 0}, {-13, 5}},
    {{-13, -7}, {-12, 12}}, {{-13, 3}, {-11, 8}},    {{-7, 12}, {-4, 7}},
    {{6, -10}, {12, 8}},    {{-9, -1}, {-7, -6}},    {{-2, -5}, {0, 12}},
    {{-12, 5}, {-7, 5}},    {{3, -10}, {8, -13}},    {{-7, -7}, {-4, 5}},
    {{-3, -2}, {-1, -7}},   {{2, 9}, {5, -11}},      {{-11, -13}, {-5, -13}},
    {{-1, 6}, {0, -1}},     {{5, -3}, {5, 2}},       {{-4, -13}, {-4, 12}},
    {{-9, -6}, {-9, 6}},    {{-12, -10}, {-8, -4}},  {{10, 2}, {12, -3}},
    {{7, 12}, {12, 12}},    {{-7, -13}, {-6, 5}},    {{-4, 9}, {-3, 4}},
    {{7, -1}, {12, 2}},     {{-7, 6}, {-5, 1}},      {{-13, 11}, {-12, 5}},
    {{-3, 7}, {-2, -6}},    {{7, -8}, {12, -7}},     {{-13, -7}, {-11, -12}},
    {{1, -3}, {12, 12}},    {{2, -6}, {3, 0}},       {{-4, 3}, {-2, -13}},
    {{-1, -13}, {1, 9}},    {{7, 1}, {8, -6}},       {{1, -1}, {3, 12}},
    {{9, 1}, {12, 6}},      {{-1, -9}, {-1, 3}},     {{-13, -13}, {-10, 5}},
    {{7, 7}, {10, 12}},     {{12, -5}, {12, 9}},     {{6, 3}, {7, 11}},
    {{5, -13}, {6, 10}},    {{2, -12}, {2, 3}},      {{3, 8}, {4, -6}},
    {{2, 6}, {12, -13}},    {{9, -12}, {10, 3}},     {{-8, 4}, {-7, 9}},
    {{-11, 12}, {-4, -6}},  {{1, 12}, {2, -8}},      {{6, -9}, {7, -4}},
    {{2, 3}, {3, -2}},      {{6, 3}, {11, 0}},       {{3, -3}, {8, -8}},
    {{7, 8}, {9, 3}},       {{-11, -5}, {-6, -4}},   {{-10, 11}, {-5, 10}},
    {{-5, -8}, {-3, 12}},   {{-10, 5}, {-9, 0}},     {{8, -1}, {12, -6}},
    {{4, -6}, {6, -11}},    {{-10, 12}, {-8, 7}},    {{4, -2}, {6, 7}},
    {{-2, 0}, {-2, 12}},    {{-5, -8}, {-5, 2}},     {{7, -6}, {10, 12}},
    {{-9, -13}, {-8, -8}},  {{-5, -13}, {-5, -2}},   {{8, -8}, {9, -13}},
    {{-9, -11}, {-9, 0}},   {{1, -8}, {1, -2}},      {{7, -4}, {9, 1}},
    {{-2, 1}, {-1, -4}},    {{11, -6}, {12, -11}},   {{-12, -9}, {-6, 4}},
    {{3, 7}, {7, 12}},      {{5, 5}, {10, 8}},       {{0, -4}, {2, 8}},
    {{-9, 12}, {-5, -13}},  {{0, 7}, {2, 12}},       {{-1, 2}, {1, 7}},
    {{5, 11}, {7, -9}},     {{3, 5}, {6, -8}},       {{-13, -4}, {-8, 9}},
    {{-5, 9}, {-3, -3}},    {{-4, -7}, {-3, -12}},   {{6, 5}, {8, 0}},
    {{-7, 6}, {-6, 12}},    {{-13, 6}, {-5, -2}},    {{1, -10}, {3, 10}},
    {{4, 1}, {8, -4}},      {{-2, -2}, {2, -13}},    {{2, -12}, {12, 12}},
    {{-2, -13}, {0, -6}},   {{4, 1}, {9, 3}},        {{-6, -10}, {-3, -5}},
    {{-3, -13}, {-1, 1}},   {{7, 5}, {12, -11}},     {{4, -2}, {5, -7}},
    {{-13, 9}, {-9, -5}},   {{7, 1}, {8, 6}},        {{7, -8}, {7, 6}},
    {{-7, -4}, {-7, 1}},    {{-8, 11}, {-7, -8}},    {{-13, 6}, {-12, -8}},
    {{2, 4}, {3, 9}},       {{10, -5}, {12, 3}},     {{-6, -5}, {-6, 7}},
    {{8, -3}, {9, -8}},     {{2, -12}, {2, 8}},      {{-11, -2}, {-10, 3}},
    {{-12, -13}, {-7, -9}}, {{-11, 0}, {-10, -5}},   {{5, -3}, {11, 8}},
    {{-2, -13}, {-1, 12}},  {{-1, -8}, {0, 9}},      {{-13, -11}, {-12, -5}},
    {{-10, -2}, {-10, 11}}, {{-3, 9}, {-2, -13}},    {{2, -3}, {3, 2}},
    {{-9, -13}, {-4, 0}},   {{-4, 6}, {-3, -10}},    {{-4, 12}, {-2, -7}},
    {{-6, -11}, {-4, 9}},   {{6, -3}, {6, 11}},      {{-13, 11}, {-5, 5}},
    {{11, 11}, {12, 6}},    {{7, -5}, {12, -2}},     {{-1, 12}, {0, 7}},
    {{-4, -8}, {-3, -2}},   {{-7, 1}, {-6, 7}},      {{-13, -12}, {-8, -13}},
    {{-7, -2}, {-6, -8}},   {{-8, 5}, {-6, -9}},     {{-5, -1}, {-4, 5}},
    {{-13, 7}, {-8, 10}},   {{1, 5}, {5, -13}},      {{1, 0}, {10, -13}},
    {{9, 12}, {10, -1}},    {{5, -8}, {10, -9}},     {{-1, 11}, {1, -13}},
    {{-9, -3}, {-6, 2}},    {{-1, -10}, {1, 12}},    {{-13, 1}, {-8, -10}},
    {{8, -11}, {10, -6}},   {{2, -13}, {3, -6}},     {{7, -13}, {12, -9}},
    {{-10, -10}, {-5, -7}}, {{-10, -8}, {-8, -13}},  {{4, -6}, {8, 5}},
    {{3, 12}, {8, -13}},    {{-4, 2}, {-3, -3}},     {{5, -13}, {10, -12}},
    {{4, -13}, {5, -1}},    {{-9, 9}, {-4, 3}},      {{0, 3}, {3, -9}},
    {{-12, 1}, {-6, 1}},    {{3, 2}, {4, -8}},       {{-10, -10}, {-10, 9}},
    {{8, -13}, {12, 12}},   {{-8, -12}, {-6, -5}},   {{2, 2}, {3, 7}},
    {{10, 6}, {11, -8}},    {{6, 8}, {8, -12}},      {{-7, 10}, {-6, 5}},
    {{-3, -9}, {-3, 9}},    {{-1, -13}, {-1, 5}},    {{-3, -7}, {-3, 4}},
    {{-8, -2}, {-8, 3}},    {{4, 2}, {12, 12}},      {{2, -5}, {3, 11}},
    {{6, -9}, {11, -13}},   {{3, -1}, {7, 12}},      {{11, -1}, {12, 4}},
    {{-3, 0}, {-3, 6}},     {{4, -11}, {4, 12}},     {{2, -4}, {2, 1}},
    {{-10, -6}, {-8, 1}},   {{-13, 7}, {-11, 1}},    {{-13, 12}, {-11, -13}},
    {{6, 0}, {11, -13}},    {{0, -1}, {1, 4}},       {{-13, 3}, {-9, -2}},
    {{-9, 8}, {-6, -3}},    {{-13, -6}, {-8, -2}},   {{5, -9}, {8, 10}},
    {{2, 7}, {3, -9}},      {{-1, -6}, {-1, -1}},    {{9, 5}, {11, -2}},
    {{11, -3}, {12, -8}},   {{3, 0}, {3, 5}},        {{-1, 4}, {0, 10}},
    {{3, -6}, {4, 5}},      {{-13, 0}, {-10, 5}},    {{5, 8}, {12, 11}},
    {{8, 9}, {9, -6}},      {{7, -4}, {8, -12}},     {{-10, 4}, {-10, 9}},
    {{7, 3}, {12, 4}},      {{9, -7}, {10, -2}},     {{7, 0}, {12, -2}},
    {{-1, -6}, {0, -11}}};

/// Calculate the rotated BRIEF descriptor bit pattern for the feature at the givven coordinates and orientation
/// within the image bitmap.
///
/// \param image    the image bitmap
/// \param coords   the coordinates of the center of the feature point within the image
/// \param angle    the rotation angle of the feature (in radian)
///
/// \return The rotated BRIEF bit pattern describing the feature point
inline Descriptor::Bits ComputeDescriptor(const gray8c_view_t& image, Point2i coords, float angle) {
    Descriptor::Bits result;

    // obtain the feature center; we use floor as rounding mode because any
    // point within a pixel square should be mapped to the pixel itself
    ptrdiff_t integer_x = static_cast<ptrdiff_t>(floorf(coords.x));
    ptrdiff_t integer_y = static_cast<ptrdiff_t>(floorf(coords.y));

    // If the feature center is too close to the image boundary to allow for accessing the 
    // required pixels, return an empty descriptor
    if (integer_x < Descriptor::kRadius || integer_x + Descriptor::kRadius >= image.width() ||
        integer_y < Descriptor::kRadius || integer_y + Descriptor::kRadius >= image.height()) {
        return result;
    }

    gray8c_view_t::xy_locator center = image.xy_at(integer_x, integer_y);

    float cos = cosf(angle), sin = sinf(angle);

    for (unsigned index = 0; index < Descriptor::kNumSamples; ++index) {
        // we are using round nearest because the center is represented by the
        // pixel center (at 1/2 pixel offset)
        ptrdiff_t first_x = 
            static_cast<ptrdiff_t>(roundf(kSampleCoordinates[index].s0.x * cos - 
                                          kSampleCoordinates[index].s0.y * sin));
        ptrdiff_t first_y = 
            static_cast<ptrdiff_t>(roundf(kSampleCoordinates[index].s0.x * sin + 
                                          kSampleCoordinates[index].s0.y * cos));
                                          
        ptrdiff_t second_x = 
            static_cast<ptrdiff_t>(roundf(kSampleCoordinates[index].s1.x * cos - 
                                          kSampleCoordinates[index].s1.y * sin));
        ptrdiff_t second_y = 
            static_cast<ptrdiff_t>(roundf(kSampleCoordinates[index].s1.x * sin + 
                                          kSampleCoordinates[index].s1.y * cos));

        result[index] = center(first_x, first_y) < center(second_x, second_y);
    }

    return result;
}

/// The kernels to use for computing the Harris scores
struct HarrisKernels {
    /// The kernel to use for calculating the gradient in x direction 
    boost::gil::detail::kernel_2d<float> dx;

    /// The kernel to use for calculating the gradient in y direction
    boost::gil::detail::kernel_2d<float> dy;

    /// The smoothing kernel to use for aggregating across tensor values
    boost::gil::detail::kernel_2d<float> smoothing;
};

/// Create the kernels that we need
///
/// \param sigma the variance of the Gaussian used for smoothing
///
/// \returns the initialized kernsels to use for calculating a Harris score
HarrisKernels CreateKernels(float sigma) {
    HarrisKernels result;

    result.dx = boost::gil::generate_dx_sobel<float>();
    result.dy = boost::gil::generate_dy_sobel<float>();

    size_t window = static_cast<size_t>(roundf(sigma * 3)) * 2 + 1;
    result.smoothing = boost::gil::generate_gaussian_kernel(window, sigma);

    return result;
}

/// Sample an image at a given location using the provided kernel
///
/// For sampling, we do not flip the kernel (as is the case for a proper convolution)
inline float Sample(const gray8c_view_t& image, Point2i coords, 
                    const boost::gil::detail::kernel_2d<float>& kernel) {
                        float result = 0;

    const auto& center_x = kernel.center_x();
    const auto& center_y = kernel.center_x();

    for (int y = -kernel.lower_size(); y <= kernel.upper_size(); ++y) {
        for (int x = -kernel.left_size(); x <= kernel.upper_size(); ++x) {
            float dx = 0, dy = 0;

            auto factor = kernel.at(x + center_x, y + center_y);
            result += factor * image(coords.x + x, coords.y + y);
        }
    }

    return result;
}

/// Compute the Harris score measuring "cornerness" for the given pixel coordinate in 
/// the image
///
/// We assume that we are dealing with a relatively small number of sample points
/// in comparison to the overall image, so we perform all calculations locally,
/// rather than performing global transformations to computer applications of
/// Gaussian filters and derivatives.
///
/// In this process as implemented here, we do not flip the kernels as would be 
/// the case for 'proper' convolutions. 
///
/// \param image    the image bitmap to use
/// \param coords   the pixel center of the candidate corner
/// \param kernels  the kernels to use for the image transformations
/// \param k        the weight `k` for the trace       
///
/// \returns        the Harris score `|M| - k trace(M)^2`, where 
///                 `M = ( I_x^2(p)    I_xy(p)
///                        I_xy(p)     I_y^2(p))`
inline float ComputeHarrisScore(const gray8c_view_t& image, Point2i coords, 
                                const HarrisKernels& kernels, float k = 0.15) {
    
    // The derivative filters should be 3x3 filters
    assert(kernels.dx.size() == 3);
    assert(kernels.dy.size() == 3);

    // the smoothing kernel should have an odd size
    assert(kernels.smoothing.size() % 2 == 1);

    double i_xx = 0, i_xy = 0, i_yy = 0;
    const auto& center_x = kernels.smoothing.center_x();
    const auto& center_y = kernels.smoothing.center_x();

    for (int y = -kernels.smoothing.lower_size(); y <= kernels.smoothing.upper_size(); ++y) {
        for (int x = -kernels.smoothing.left_size(); x <= kernels.smoothing.upper_size(); ++x) {
            Point2i sample_coords(coords.x + x, coords.y + y);
            float dx = Sample(image, sample_coords, kernels.dx);
            float dy = Sample(image, sample_coords, kernels.dy);

            auto factor = kernels.smoothing.at(center_x + x, center_y + y);
            i_xx += dx * dx * factor;
            i_xy += dx * dy * factor;
            i_yy += dy * dy * factor;
        }
    }

    double det = i_xx * i_yy - i_xy * i_xy;
    return static_cast<float>(det - k * (i_xx + i_yy));
}

/// Convert an RGB image to a grayscale image
///
/// Note: This function is lifted from an example in the Boost GIL source tree and subject 
/// to the BOOST 1.0 license
/// 
/// Copyright 2019 Olzhas Zhumabek <anonymous.from.applecity@gmail.com>
///
/// \param original the image to convert
///
/// \return a grayscale image with same dimensions and resolution as the original
boost::gil::gray8_image_t RgbToGrayscale(const boost::gil::rgb8_view_t& original) {
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

class Bitmap {
public:
    Bitmap(size_t width, size_t height)
        : width_(width), height_(height), bits_(width * height) {}

    void Clear() {
        std::fill(bits_.begin(), bits_.end(), false);
    }

    void Set(size_t x, size_t y) {
        size_t index = x * y * width_;
        bits_[index] = true;
    }

    bool Get(size_t x, size_t y) const {
        size_t index = x * y * width_;
        return bits_[index];
    }

    size_t width() const { return width_; }
    size_t height() const { return height_; }

private:
    size_t width_, height_;
    std::vector<bool> bits_;
};

/// The parameters needed to fully specify ORB feature detection and descriptor
struct OrbParameters {
    /// The maximum number of features to return
    unsigned max_features;

    /// Minimum distance between returned feature points
    int min_distance = 10;

    /// Number of levels in the image pyramid
    unsigned levels = 4;

    /// Scale factor between levels in the pyramid
    float scale_factor = sqrtf(0.5f);

    /// FAST detection threshold
    float threshold = 20;

    /// window size for calculating orientation
    int window_size = 7;
    
    /// Variance (sigma) of smoothing filter
    float sigma = 1.0;

    /// `k`-parameter in Harris score
    float k = 0.15;
};

/// Detect ORB features and compute their descriptor values
///
/// \param original the original RGB image in which we want to detect feature points
/// \param parameters the parameter set to apply during the various stages of the computation
///
/// \returns the collection of detected features and their description
std::vector<Descriptor> ComputeFeatures(const boost::gil::rgb8_view_t& original, const OrbParameters& parameters) {    
    std::vector<Descriptor> result;

    auto level_0 = RgbToGrayscale(original);
    auto pyramid = CreatePyramid(std::move(level_0), parameters.scale_factor, parameters.levels);

    // For each level:
    //  detect feature points and calculate their Harris score
    //  determine the points with the highest score
    //  fill in the decriptor using the image at the appropriate level

    struct Candidate {
        Point2i     coords;
        unsigned    level;
        float       harris_score;
    };

    std::vector<Candidate> candidates;
    auto kernels = CreateKernels(parameters.sigma);
    float scale_factor = 1.0f;
    std::vector<float> pyramid_factor;

    for (unsigned level = 0; level < parameters.levels; ++level, scale_factor *= parameters.scale_factor) {
        pyramid_factor.push_back(1.0f/scale_factor);

        auto image_view = boost::gil::const_view(pyramid[level]);
        // TODO: Determine a border width to exclude feature points being generated too close to
        // the boundary of the image. This needs to reflect the mask/kernel sizes of all the processing
        // stages
        auto points = DetectFastFeatures(image_view, parameters.threshold);

        for (const auto& point: points) {
            candidates.emplace_back(Candidate { 
                point, level, ComputeHarrisScore(image_view, point, kernels, parameters.k)
            });
        }
    }

    // Sort features in descending order by Harris score
    std::sort(candidates.begin(), candidates.end(), [](const auto& left, const auto& right) {
        return left.harris_score > right.harris_score;
    });

    // We are usng a simple bitmap to perform surpression of nearby features
    float grid_size = parameters.min_distance * 0.5f;
    float inv_grid_size = 1.0f / grid_size;
    size_t mask_width = ceilf(original.width() * inv_grid_size);
    size_t mask_height = ceilf(original.height() * inv_grid_size);

    Bitmap mask(mask_width, mask_height);

    // Select the top N features and create the result
    for (const auto& feature: candidates) {
        if (result.size() >= parameters.max_features) {
            break;
        }

        auto scale = pyramid_factor[feature.level] * inv_grid_size;
        Point2f coord(feature.coords.x * scale, feature.coords.y * scale);
        int x = floorf(coord.x), y = floorf(coord.y);

        if (mask.Get(x, y)) {
            continue;
        }

        int x_left = std::max(0, x - 1);
        int x_right = std::min((int) mask_width - 1, x + 1);
        int y_up = std::max(0, y - 1);
        int y_down = std::min((int) mask_height - 1, y + 1);

        mask.Set(x, y);
        mask.Set(x_left, y);
        mask.Set(x_right, y);
        mask.Set(x, y_up);
        mask.Set(x_left, y_up);
        mask.Set(x_right, y_up);
        mask.Set(x, y_down);
        mask.Set(x_left, y_down);
        mask.Set(x_right, y_down);

        auto image = const_view(pyramid[feature.level]);
        float angle = CalculateOrientation(image, feature.coords, parameters.window_size/2);
        result.emplace_back(Descriptor {
            coord, angle, feature.level, ComputeDescriptor(image, feature.coords, angle)
        });
    }

    return result;
}

} // namespace