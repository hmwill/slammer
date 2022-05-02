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

#include "slammer/poses_locations_optimizer.h"

#include "slammer/optimizer.h"

using namespace slammer;

namespace {

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

Result<double> PosesLocationsOptimizer::Optimize(Poses& poses, Locations& locations, bool inout, const Parameters& parameters) {
    assert(poses.size() == keyframes_.size());
    assert(locations.size() == landmarks_.size());

    Eigen::VectorXd value(total_dimension());

    // Initialize value vector
    // TODO: Should we do something smart in the case of mapped landmarks?
    // Such as initializing using the mean of the locations of the unified landmarks?
    if (inout) {
        for (size_t index = 0; index < poses.size(); ++index) {
            auto slot = PoseSlot(index);
            value(slot) = poses[index].log();
        }

        for (size_t index = 0; index < locations.size(); ++index) {
            auto slot = LocationSlot(index);
            value(slot) = locations[index];
        }
    } else {
        for (size_t index = 0; index < keyframes_.size(); ++index) {
            auto slot = PoseSlot(index);
            value(slot) = keyframes_[index]->pose.log();
        }

        for (size_t index = 0; index < landmarks_.size(); ++index) {
            auto slot = LocationSlot(index);
            value(slot) = landmarks_[index]->location;
        }
    }

    // perform optimization
    using namespace std::placeholders;

    auto result = 
        LevenbergMarquardt(std::bind(&PosesLocationsOptimizer::CalculateJacobian, *this, _1),
                            std::bind(&PosesLocationsOptimizer::CalculateResidual, *this, _1),
                            value, parameters.max_iterations, parameters.lambda);

    // extract result
    for (size_t index = 0; index < poses.size(); ++index) {
        auto slot = PoseSlot(index);
        poses[index] = SE3d::exp(value(slot));
    }

    // for landmarks, reflect the provided mapping of landmarks when extracting result values
    for (size_t index = 0; index < locations.size(); ++index) {
        size_t slot_index = RemapLandmarkIndex(index);

        auto slot = LocationSlot(slot_index);
        locations[index] = value(slot);
    }

    return result;
}

Eigen::SparseMatrix<double> PosesLocationsOptimizer::CalculateJacobian(const Eigen::VectorXd& value) const {
    assert(value.size() == total_dimension());

    using Triplet = Eigen::Triplet<double>;
    std::vector<Triplet> triplets;
    size_t num_constraints = 0;

    for (size_t index = 0; index < landmarks_.size(); ++index) {
        // we use the landmark as is to identify constraints
        const Landmark& landmark = *landmarks_[index];

        // but we use the remapped coordinates for solving the equation
        size_t location_index = RemapLandmarkIndex(index);
        auto location_slot = LocationSlot(location_index);

        Vector3d coords(value(location_slot));

        for (const auto& feature: landmark.observations) {
            auto keyframe_pointer = feature.lock()->keyframe.lock();
            auto keyframe_lookup = keyframe_index_.find(keyframe_pointer);

            if (keyframe_lookup != keyframe_index_.end()) {
                // the keyframe pose is also subject to optimization
                size_t keyframe_index = keyframe_lookup->second;
                auto pose_slot = PoseSlot(keyframe_index);

                SE3d::Tangent tangent = value(pose_slot);
                SE3d pose = SE3d::exp(tangent);

                // generate entries for second measurement error depending on second measurement
                Point3d transformed = pose * coords;
                Matrix3d jacobian_project = depth_camera_.Jacobian(transformed);
                Matrix3d jacobian_coords = jacobian_project * pose.so3().matrix();

                EmitTriplets(jacobian_coords, std::back_insert_iterator(triplets), num_constraints, location_slot.first());

                // generate entries for second measurement error depending on second pose
                Eigen::Matrix<double, 3, 6> jacobian_second_params;
                jacobian_second_params(Eigen::seqN(0, 3), Eigen::seqN(0, 3)) = Matrix3d::Identity();
                jacobian_second_params(Eigen::seqN(0, 3), Eigen::seqN(3, 3)) = -SE3d::SO3Member::hat(transformed);
                Eigen::Matrix<double, 3, 6> jacobian_params = jacobian_project * jacobian_second_params;

                EmitTriplets(jacobian_params, std::back_insert_iterator(triplets), num_constraints, pose_slot.first());
            } else {
                // the keyframe pose is fixed
                const SE3d& pose = keyframe_pointer->pose;

                // generate entries for second measurement error depending on second measurement
                Point3d transformed = pose * coords;
                Matrix3d jacobian_project = depth_camera_.Jacobian(transformed);
                Matrix3d jacobian_coords = jacobian_project * pose.so3().matrix();

                EmitTriplets(jacobian_coords, std::back_insert_iterator(triplets), num_constraints, location_slot.first());
            }

            num_constraints += kDimConstraint;
        }
    }

    for (size_t index = 0; index < keyframes_.size(); ++index) {
        const Keyframe& keyframe = *keyframes_[index];
        auto pose_slot = PoseSlot(index);

        SE3d::Tangent tangent = value(pose_slot);
        SE3d pose = SE3d::exp(tangent);

        for (const auto& feature: keyframe.features) {
            auto landmark_pointer = feature->landmark.lock();

            // skip landmarks that we processed in the loop above
            if (landmark_id_index_.find(landmark_pointer->id) != landmark_id_index_.end()) {
                continue;
            }

            // Constant landmarks cannot be subject to unification
            assert(mapping_.find(landmark_pointer->id) == mapping_.end());

            // we are only dealing with constraints given by landmarks whose
            // position we consider as fixed
            const Vector3d& coords = landmark_pointer->location;

            // generate entries for second measurement error depending on second measurement
            Point3d transformed = pose * coords;
            Matrix3d jacobian_project = depth_camera_.Jacobian(transformed);

            // generate entries for second measurement error depending on second pose
            Eigen::Matrix<double, 3, 6> jacobian_second_params;
            jacobian_second_params(Eigen::seqN(0, 3), Eigen::seqN(0, 3)) = Matrix3d::Identity();
            jacobian_second_params(Eigen::seqN(0, 3), Eigen::seqN(3, 3)) = -SE3d::SO3Member::hat(transformed);
            Eigen::Matrix<double, 3, 6> jacobian_params = jacobian_project * jacobian_second_params;

            EmitTriplets(jacobian_params, std::back_insert_iterator(triplets), num_constraints, pose_slot.first());

            num_constraints += kDimConstraint;
        }
    }

    assert(num_constraints == total_constraints());

    Eigen::SparseMatrix<double> result(total_constraints(), total_dimension());
    result.setFromTriplets(triplets.begin(), triplets.end());
    return result;
}

Eigen::VectorXd PosesLocationsOptimizer::CalculateResidual(const Eigen::VectorXd& value) const {
    assert(value.size() == total_dimension());
    Eigen::VectorXd residual(total_constraints());

    size_t num_constraints = 0;

    for (size_t index = 0; index < landmarks_.size(); ++index) {
        const Landmark& landmark = *landmarks_[index];

        // but we use the remapped coordinates for solving the equation
        size_t location_index = RemapLandmarkIndex(index);
        auto location_slot = LocationSlot(location_index);

        for (const auto& weak_feature: landmark.observations) {
            auto feature = weak_feature.lock();
            auto keyframe_pointer = feature->keyframe.lock();
            auto keyframe_lookup = keyframe_index_.find(keyframe_pointer);
            auto residual_slot = Eigen::seqN(num_constraints, int(kDimConstraint));

            SE3d pose;

            if (keyframe_lookup != keyframe_index_.end()) {
                // the keyframe pose is also subject to optimization
                size_t keyframe_index = keyframe_lookup->second;
                auto pose_slot = PoseSlot(keyframe_index);
                SE3d::Tangent tangent = value(pose_slot);
                pose = SE3d::exp(tangent);
            } else {
                pose = keyframe_pointer->pose;
            }

            Point3d where = value(location_slot);
            auto coords = pose * where;
            auto projection = depth_camera_.CameraToPixelDisparity(coords);
            residual(residual_slot) = projection - depth_camera_.CameraToPixelDisparity(feature->coords);

            num_constraints += kDimConstraint;
        }
    }

    for (size_t index = 0; index < keyframes_.size(); ++index) {
        const Keyframe& keyframe = *keyframes_[index];
        auto pose_slot = PoseSlot(index);
        SE3d pose = SE3d::exp(value(pose_slot));

        for (const auto& feature: keyframe.features) {
            auto landmark_pointer = feature->landmark.lock();

            // skip landmarks that we processed in the loop above
            if (landmark_id_index_.find(landmark_pointer->id) != landmark_id_index_.end()) {
                continue;
            }

            // Constant landmarks cannot be subject to unification
            assert(mapping_.find(landmark_pointer->id) == mapping_.end());

            // we are only dealing with constraints given by landmarks whose
            // position we consider as fixed
            auto residual_slot = Eigen::seqN(num_constraints, int(kDimConstraint));

            auto coords = pose * landmark_pointer->location;
            auto projection = depth_camera_.CameraToPixelDisparity(coords);
            residual(residual_slot) = projection - depth_camera_.CameraToPixelDisparity(feature->coords);

            num_constraints += kDimConstraint;
        }
    }

    assert(num_constraints == total_constraints());
    return residual;
}

void PosesLocationsOptimizer::CalculateConstraints() {
    size_t num_constraints = 0;

    for (size_t index = 0; index < landmarks_.size(); ++index) {
        const Landmark& landmark = *landmarks_[index];

        for (const auto& feature: landmark.observations) {
            num_constraints += kDimConstraint;
        }
    }

    for (size_t index = 0; index < keyframes_.size(); ++index) {
        const Keyframe& keyframe = *keyframes_[index];

        for (const auto& feature: keyframe.features) {
            auto landmark_pointer = feature->landmark.lock();

            // skip landmarks that we processed in the loop above
            if (landmark_id_index_.find(landmark_pointer->id) != landmark_id_index_.end()) {
                continue;
            }

            num_constraints += kDimConstraint;
        }
    }

    total_constraints_ = num_constraints;
}