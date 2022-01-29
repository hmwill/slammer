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

#include "slammer/flow.h"

#include <boost/gil/extension/numeric/sampler.hpp>
//#include <boost/gil/extension/numeric/resample.hpp>

#include "Eigen/Dense"

using namespace slammer;

using namespace boost::gil;

#define AFFINE_VERSION 1

namespace {

gray32f_image_t ComputeDx(const gray8c_view_t& image) {
    // could really just be a Scharr kernel and 1D convolution
    auto kernel_dx = boost::gil::generate_dx_sobel<float>();
    //static const auto kernel_dx = boost::gil::generate_dx_scharr<float>();

    auto sum = std::reduce(kernel_dx.begin(), kernel_dx.end(), 0.0f, 
                             [](auto left, auto right) { return left + fabsf(right); });
    for (auto& val: kernel_dx) {
        val /= sum;
    }

    gray32f_image_t result(image.dimensions());
    boost::gil::detail::convolve_2d(image, kernel_dx, view(result));

    return result;
}

gray32f_image_t ComputeDy(const gray8c_view_t& image) {
    // could really just be a Scharr kernel and 1D convolution
    auto kernel_dy = boost::gil::generate_dy_sobel<float>();
    //static const auto kernel_dy = boost::gil::generate_dy_scharr<float>();

    auto sum = std::reduce(kernel_dy.begin(), kernel_dy.end(), 0.0f, 
                             [](auto left, auto right) { return left + fabsf(right); });
    for (auto& val: kernel_dy) {
        val /= sum;
    }
    gray32f_image_t result(image.dimensions());
    boost::gil::detail::convolve_2d(image, kernel_dy, view(result));

    return result;
}

using Vec2 = Eigen::Vector2f;
using Mat2 = Eigen::Matrix2f;

#if AFFINE_VERSION
using Rhs = Eigen::Matrix<double, 6, 1>;
using Lhs = Eigen::Matrix<double, 6, 6>;
#else
using Rhs = Eigen::Matrix<double, 2, 1>;
using Lhs = Eigen::Matrix<double, 2, 2>;
#endif

Lhs CalculateG(const gray32fc_view_t& image_dx, const gray32fc_view_t& image_dy,
                const Vec2& x, unsigned omega) {
    float omegaf = static_cast<float>(omega);
    Lhs G = Lhs::Zero();

    for (float dx = -omegaf; dx <= omegaf; dx += 1.0f) {
        for (float dy = -omegaf; dy <= omegaf; dy += 1.0f) {
            gray32f_pixel_t ix, iy;
            auto pos = point2<float>(x.x() + dx, x.y() + dy);
            sample(bilinear_sampler{}, image_dx, pos, ix); 
            sample(bilinear_sampler{}, image_dy, pos, iy); 

            Rhs vec;
            vec << ix[0], iy[0]
#if AFFINE_VERSION
            , dx * ix[0], dx * iy[0], dy * ix[0], dy * iy[0]
            ;
#endif

            G += vec * vec.transpose();
        }
    }

    return G;
}

Rhs CalcDiff(const gray8c_view_t& source, const gray8c_view_t& target, 
              const gray32fc_view_t& image_dx, const gray32fc_view_t& image_dy,
              const Mat2& A, const Vec2& v, const Vec2& x, unsigned omega) {
    float omegaf = static_cast<float>(omega);
    Rhs diff = Rhs::Zero();

    for (float dx = -omegaf; dx <= omegaf; dx += 1.0f) {
        for (float dy = -omegaf; dy <= omegaf; dy += 1.0f) {
            gray8_pixel_t source_pixel, target_pixel;
            auto pos = point2<float>(x.x() + dx, x.y() + dy);
            sample(bilinear_sampler{}, source, pos, source_pixel); 

            auto transformed_v = A * Vec2(dx, dy) + v + x; 
            auto target_pos = point2<float>(transformed_v[0], transformed_v[1]);
            sample(bilinear_sampler{}, target, target_pos, target_pixel); 
            auto delta = (float) source_pixel[0] - (float) target_pixel[0];

            gray32f_pixel_t ix, iy;
            sample(bilinear_sampler{}, image_dx, pos, ix); 
            sample(bilinear_sampler{}, image_dy, pos, iy); 

            Rhs vec;
            vec << ix[0] * delta, 
                iy[0] * delta
#if AFFINE_VERSION
                , 
                dx * ix[0] * delta, 
                dx * iy[0] * delta, 
                dy * ix[0] * delta, 
                dy * iy[0] * delta
#endif
                ;

            diff += vec;
        }
    }

    return diff;
}

} // namespace

void 
slammer::ComputeFlow(const gray8c_view_t& source, const gray8c_view_t& target,
                     const std::vector<Point2f>& source_points, std::vector<Point2f>& target_points,
                     std::vector<float>& error, unsigned num_levels, unsigned omega, float threshold) {
    // calculate image pyramids
    std::vector<gray8_image_t> source_pyramid, target_pyramid;
    AppendPyramidLevels(source, 0.5f, num_levels, source_pyramid);
    AppendPyramidLevels(target, 0.5f, num_levels, target_pyramid);
    std::vector<gray8c_view_t> view_source_pyramid, view_target_pyramid;
    view_source_pyramid.push_back(source);
    view_target_pyramid.push_back(target);
    std::transform(source_pyramid.begin(), source_pyramid.end(), std::back_inserter(view_source_pyramid),
                   [](const auto& image) { return const_view(image); });
    std::transform(target_pyramid.begin(), target_pyramid.end(), std::back_inserter(view_target_pyramid),
                   [](const auto& image) { return const_view(image); });

    // create derivates for each level of the pyramid
    std::vector<gray32f_image_t> source_pyramid_dx, source_pyramid_dy;
    std::transform(view_source_pyramid.begin(), view_source_pyramid.end(), std::back_inserter(source_pyramid_dx),
                   ComputeDx);
    std::transform(view_source_pyramid.begin(), view_source_pyramid.end(), std::back_inserter(source_pyramid_dy),
                   ComputeDy);

    // initialize target coodinates
    size_t num_points = source_points.size();

    if (target_points.empty()) {
        target_points = source_points;
    } else {
        assert(target_points.size() == num_points);
    }

    error.clear();
    error.resize(num_points, std::numeric_limits<float>::quiet_NaN());

    // iterate from highest down to lowest pyramid level
    for (size_t index = 0; index < num_points; ++index) {
        float err = std::numeric_limits<float>::max();

        float scale = 1.0f / (1 << (num_levels + 1));
        Vec2 x = Vec2(source_points[index].x, source_points[index].y) * scale;
        Vec2 v = Vec2(target_points[index].x, target_points[index].y) * scale;
        v -= x;
        Mat2 A = Mat2::Identity();

        for (unsigned level = num_levels + 1; level;) {
            --level;
            v *= 2.0f;
            x *= 2.0f;
            err = std::numeric_limits<float>::max();

            const auto& image_dx = const_view(source_pyramid_dx[level]);
            const auto& image_dy = const_view(source_pyramid_dy[level]);
            const auto& source = view_source_pyramid[level];
            const auto& target = view_target_pyramid[level];

            Lhs G = CalculateG(image_dx, image_dy, x, omega);
            Eigen::LLT<decltype(G)> decomposition(G);

            // iterate until convergence
            unsigned remaining_iterations = 10;

            do {
                Rhs diff = CalcDiff(source, target, image_dx, image_dy, A, v, x, omega);
                Rhs step = decomposition.solve(diff);

                auto previous_err = err;
                Vec2 eta_xy(step[0], step[1]);
                err = eta_xy.squaredNorm();

                if (err >= previous_err || err < threshold * threshold) {
                    break;
                }

                v += A * eta_xy;

#if AFFINE_VERSION
                Mat2 update;
                update <<
                    step[2] + 1.0f, step[3],        
                    step[4],        step[5] + 1.0f;

                A = A * update;
#endif
            } while (remaining_iterations--);
        }

        target_points[index] = source_points[index] + Point2f(v[0], v[1]);
        error[index] = err;
    }
}