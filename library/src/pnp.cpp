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

#include "slammer/pnp.h"

#include "slammer/optimizer.h"

using namespace slammer;

namespace {
    
void CreateRandomIndices(std::vector<size_t>& indices, std::default_random_engine& random_engine) {
    std::iota(begin(indices), end(indices), 0);
    std::shuffle(begin(indices), end(indices), random_engine);
}

template <class Element>
std::vector<Element> Subset(const std::vector<Element>& all, const std::vector<size_t>& indices, size_t sample_size) {
    std::vector<Element> result;

    for (size_t index = 0; index < sample_size; ++index) {
        result.push_back(all[indices[index]]);
    }

    return result;
}

template <typename Matrix, typename Iterator>
void EmitTriplets(const Matrix& matrix, Iterator output, size_t row_offset, size_t column_offset) {
    for (size_t row_index = 0; row_index < matrix.rows(); ++row_index) {
        for (size_t column_index = 0; column_index < matrix.cols(); ++column_index) {
            auto value = matrix(row_index, column_index);

            if (value) {
                *output++ = Eigen::Triplet<double>(row_offset + row_index, column_offset + column_index, value);
            }
        }
    }
}

} // namespace

Eigen::SparseMatrix<double> 
PerspectiveAndPoint3d::CalculateJacobian(const Eigen::VectorXd& value) const {
    SE3d::Tangent tangent(value(Eigen::seqN(0, 6)));
    SE3d second_pose = SE3d::exp(tangent);

    Eigen::Matrix<double, 3, 3> jacobian_second_coords = second_pose.so3().matrix();

    const size_t rows = point_pairs_.size() * 6;
    const size_t columns = point_pairs_.size() * 3 + 6;
    using Triplet = Eigen::Triplet<double>;
    std::vector<Triplet> triplets;

    // fill in the coefficients of the matrix
    for (size_t index = 0; index < point_pairs_.size(); ++index) {
        Vector3d coords(value(Eigen::seqN(index * 3 + 6, 3)));

        // generate entries for first measurement error depending on first measurement
        // this is just the Jacobian of mapping the point coordinates via the camera
        // to pixel coordinates.
        auto jacobian = first_.Jacobian(coords);

        EmitTriplets(jacobian, std::back_insert_iterator(triplets), index * 6, index * 3 + 6);

        // generate entries for second measurement error depending on second measurement
        Point3d transformed = second_pose * coords;
        Matrix3d jacobian_project = second_.Jacobian(transformed);
        Matrix3d jacobian_coords = jacobian_project * second_pose.so3().matrix();

        EmitTriplets(jacobian_coords, std::back_insert_iterator(triplets), index * 6 + 3, index * 3 + 6);

        // generate entries for second measurement error depending on second pose
        Eigen::Matrix<double, 3, 6> jacobian_second_params;
        jacobian_second_params(Eigen::seqN(0, 3), Eigen::seqN(0, 3)) = Matrix3d::Identity();
        jacobian_second_params(Eigen::seqN(0, 3), Eigen::seqN(3, 3)) = -SE3d::SO3Member::hat(transformed);
        Eigen::Matrix<double, 3, 6> jacobian_params = jacobian_project * jacobian_second_params;

        EmitTriplets(jacobian_params, std::back_insert_iterator(triplets), index * 6 + 3, 0);
    }

    Eigen::SparseMatrix<double> result(rows, columns);
    result.setFromTriplets(triplets.begin(), triplets.end());
    return result;
}

Eigen::VectorXd 
PerspectiveAndPoint3d::CalculateResidual(const Eigen::VectorXd& value) const {
    SE3d::Tangent tangent(value(Eigen::seqN(0, 6)));
    SE3d second_pose = SE3d::exp(tangent);

    Eigen::VectorXd residual(point_pairs_.size() * 6);

    for (size_t index = 0; index < point_pairs_.size(); ++index) {
        Vector3d coords(value(Eigen::seqN(index * 3 + 6, 3)));
        auto first_projection = first_.CameraToPixelDisparity(coords);
        residual(Eigen::seqN(index * 6, 3)) = first_projection - point_pairs_[index].first;

        auto second_coords = second_pose * coords;
        auto second_projection = second_.CameraToPixelDisparity(second_coords);
        residual(Eigen::seqN(index * 6 + 3, 3)) = second_projection - point_pairs_[index].second;
    }

    return residual;
}

Result<double> 
PerspectiveAndPoint3d::SolveLevenberg(SE3d& pose, std::vector<Point3d>& points, unsigned max_iterations, 
                                      double lambda) {
    // Reconstruct pose and locations; using true locations and identity pose as starting point
    Eigen::VectorXd variables(points.size() * 3 + 6);

    for (size_t index = 0; index < points.size(); ++index) {
        variables(Eigen::seqN(index * 3 + 6, 3)) = points[index];
    }

    variables(Eigen::seqN(0, 6)) = pose.log();

    using namespace std::placeholders;

    auto result = 
        LevenbergMarquardt(std::bind(&PerspectiveAndPoint3d::CalculateJacobian, *this, _1),
                            std::bind(&PerspectiveAndPoint3d::CalculateResidual, *this, _1),
                            variables, max_iterations, lambda);

    if (result.ok()) {
        // extract the optimization result
        pose = SE3d::exp(variables(Eigen::seqN(0, 6)));

        for (size_t index = 0; index < points.size(); ++index) {
            points[index] = variables(Eigen::seqN(index * 3 + 6, 3));
        }
    }

    return result;
}

Result<double> 
PerspectiveAndPoint3d::Ransac(SE3d& pose, std::vector<uchar>& inlier_mask, const std::vector<Point3d>& points, 
                              size_t sample_size, unsigned max_iterations, double lambda, double threshold,
                            std::default_random_engine& random_engine) {
    std::vector<size_t> indices(points.size());

    unsigned max_additional_inliers = 0;
    std::vector<size_t> max_inlier_indices, candidate_inlier_indices;

    for (unsigned iter = max_iterations; iter--;) {
        // Create a random index permutation; the prefix of sample_size elements will serve as sample
        CreateRandomIndices(indices, random_engine);
        auto sample_points = Subset(points, indices, sample_size);
        auto sample_point_pairs = Subset(point_pairs_, indices, sample_size);

        // Create an instance based on the sample set
        PerspectiveAndPoint3d instance(first_, second_, sample_point_pairs);

        // Calculate the solution for the sample instance
        SE3d instance_pose = pose;
        auto result = instance.SolveLevenberg(instance_pose, sample_points, max_iterations, lambda);

        if (!result.ok()) {
            // didn't yield a solvable instance
            continue;
        }

        // Determine the number of inliers not countained in the sample set; here we need to back project
        // the first meassurement into world space, then transform it via the computed camera pose and compare
        // the error against the second measurement
        unsigned num_additional_inliers = 0;
        candidate_inlier_indices.clear();

        for (size_t index = 0; index < points.size(); ++index) {
            const auto& point_pair = point_pairs_[indices[index]];
            auto prediction = 
                second_.CameraToPixelDisparity(instance_pose *
                                                first_.PixelDisparityToCamera(point_pair.first));

            double error = (point_pair.second - prediction).squaredNorm();

            if (index < sample_size) {
                if (error > threshold) {
                    break;
                }
            } else {
                if (error <= threshold) {
                    num_additional_inliers++;
                    candidate_inlier_indices.push_back(indices[index]);
                }
            }
        }

        if (num_additional_inliers && num_additional_inliers > max_additional_inliers) {
            max_additional_inliers = num_additional_inliers;
            max_inlier_indices = candidate_inlier_indices;

            std::copy(indices.begin(), indices.begin() + sample_size, std::back_insert_iterator(max_inlier_indices));
        }
    }

    inlier_mask.clear();

    if (max_additional_inliers) {
        inlier_mask.resize(points.size(), 0);

        for (auto index: max_inlier_indices) {
            inlier_mask[index] = ~0;
        }

        auto robust_points = Subset(points, max_inlier_indices, max_inlier_indices.size());
        auto robust_point_pairs = Subset(point_pairs_, max_inlier_indices, max_inlier_indices.size());
        PerspectiveAndPoint3d instance(first_, second_, robust_point_pairs);

        return instance.SolveLevenberg(pose, robust_points, max_iterations, lambda);
    } else {
        inlier_mask.resize(points.size(), ~0);
        PerspectiveAndPoint3d instance(first_, second_, point_pairs_);
        std::vector<Point3d> temp_points(points);
        return instance.SolveLevenberg(pose, temp_points, max_iterations, lambda);
    }
}
