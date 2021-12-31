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

#include "slammer/math.h"

#include <algorithm>
#include <numeric>

#include "Eigen/SVD"


using namespace slammer;


namespace {

void CreateRandomIndices(std::vector<size_t>& indices, std::default_random_engine& random_engine) {
    std::iota(begin(indices), end(indices), 0);
    std::shuffle(begin(indices), end(indices), random_engine);
}

double EstimateVariance(const std::vector<Point3d>& reference, const std::vector<Point3d>& transformed,
                        const SE3d& transformation) {
    double sum = 0;
    auto size = reference.size();

    for (size_t index = 0; index < size; ++index) {
        Point3d diff = transformed[index] - transformation * reference[index];
        sum += diff.squaredNorm();
    }

    return sum / (size - 1);
}

} // namespace


SE3d slammer::CalculateIcp(const std::vector<Point3d>& reference, const std::vector<Point3d>& transformed) {
    using std::begin, std::end;

    size_t num_elements = reference.size();
    assert(transformed.size() == num_elements);
    double factor = 1.0 / num_elements;

    // Calculate the centroids of the two point clouds
    Point3d centroid_reference = std::accumulate(begin(reference), end(reference), Point3d(0.0, 0.0, 0.0)) * factor;
    Point3d centroid_transformed = std::accumulate(begin(transformed), end(transformed), Point3d(0.0, 0.0, 0.0)) * factor;

    // Calculate adjusted coordinates for each point cloud, where the centroid has been moved to the origin
    std::vector<Eigen::Vector3d> shifted_reference, shifted_transformed;
    shifted_reference.reserve(num_elements);
    shifted_transformed.reserve(num_elements);

    std::transform(begin(reference), end(reference), std::back_inserter(shifted_reference),
                   [&](const Point3d& coord) { return coord - centroid_reference; });
    std::transform(begin(transformed), end(transformed), std::back_inserter(shifted_transformed),
                   [&](const Point3d& coord) { return coord - centroid_transformed; });

    // Calculate the error matrix for point rotations
    Eigen::Matrix3d W = Eigen::Matrix3d::Zero();

    for (size_t index = 0; index < num_elements; ++index) {
        W += shifted_transformed[index] * shifted_reference[index].transpose();
    }

    // Compute the SVD
    Eigen::JacobiSVD<Eigen::Matrix3d> svd(W, Eigen::ComputeFullU | Eigen::ComputeFullV);
    const auto& U = svd.matrixU();
    const auto& V = svd.matrixV();

    // Extract the transformation components
    Eigen::Matrix3d rotation = U * V.transpose();

    if (rotation.determinant() < 0) {
        rotation = -rotation;
    }

    Eigen::Vector3d translation = centroid_transformed - rotation * centroid_reference;
    return SE3d(rotation, translation);
}

size_t slammer::RobustIcp(const std::vector<Point3d>& reference, const std::vector<Point3d>& transformed,
                          std::default_random_engine& random_engine,
                          SE3d& best_estimate, std::vector<uchar>& inlier_mask,
                          size_t max_iterations, size_t sample_size, double threshold,
                          size_t min_additional_inliers) {
    size_t num_points = reference.size();
    assert(transformed.size() == num_points);

    inlier_mask.clear();
    size_t best_num_inliers = 0;
    double best_variance = std::numeric_limits<double>::max();
    inlier_mask.resize(num_points, 0);

    if (num_points > sample_size + min_additional_inliers) {
        std::vector<size_t> indices(num_points);
        std::vector<uchar> inliers(num_points);
        std::vector<Point3d> sample_reference, sample_transformed;
        sample_reference.reserve(sample_size);
        sample_transformed.reserve(sample_size);

        for (size_t num_iteration = 0; num_iteration < max_iterations; ++num_iteration) {
            CreateRandomIndices(indices, random_engine);
            sample_reference.clear();
            sample_transformed.clear();
            std::fill(inliers.begin(), inliers.end(), 0);

            for (size_t index = 0; index < sample_size; ++index) {
                sample_reference.push_back(reference[indices[index]]);
                sample_transformed.push_back(transformed[indices[index]]);
                inliers[indices[index]] = std::numeric_limits<uchar>::max();
            }

            SE3d estimate = CalculateIcp(sample_reference, sample_transformed);
            size_t num_additonal_inliers = 0;

            for (size_t index = 0; index < num_points; ++index) {
                if (inliers[index]) {
                    continue;
                }

                Point3d diff = transformed[index] - estimate * reference[index];
                double squared_distance = diff.squaredNorm();

                bool is_inlier = squared_distance < threshold;
                inliers[index] = is_inlier ? std::numeric_limits<uchar>::max() : 0;
                num_additonal_inliers += is_inlier;
            }

            if (num_additonal_inliers >= min_additional_inliers) {
                sample_reference.clear();
                sample_transformed.clear();

                for (size_t index = 0; index < num_points; ++index) {
                    if (inliers[index]) {
                        sample_reference.push_back(reference[index]);
                        sample_transformed.push_back(transformed[index]);
                    }
                }

                SE3d estimate = CalculateIcp(sample_reference, sample_transformed);
                double variance = EstimateVariance(sample_reference, sample_transformed, estimate);

                if (variance < best_variance) {
                    best_variance = variance;
                    best_num_inliers = num_additonal_inliers + sample_size;
                    best_estimate = estimate;
                    inlier_mask = inliers;
                }
            }
        }
    }

    if (!best_num_inliers) {
        best_estimate = CalculateIcp(reference, transformed);
        inlier_mask.resize(num_points, std::numeric_limits<uchar>::max());
    }

    return best_num_inliers;
}