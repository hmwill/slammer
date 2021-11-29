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

#include "slammer/backend.h"

#include "slammer/map.h"

#include "g2o/types/slam3d/types_slam3d.h"
#include "g2o/core/sparse_optimizer.h"
#include "g2o/core/block_solver.h"
#include "g2o/core/robust_kernel.h"
#include "g2o/core/robust_kernel_impl.h"
#include "g2o/core/optimization_algorithm_levenberg.h"
#include "g2o/solvers/csparse/linear_solver_csparse.h"

using namespace slammer;

namespace {

// Link all features in the incoming new frame that have been matched against a given reference
// frame to the landmarks already associated with their corresponding feature.
void MatchFeaturesToLandmarks(KeyframePointer& reference_frame, KeyframePointer& new_frame,
                              const std::vector<cv::DMatch>& matches) {
    for (const auto& match: matches) {
        auto landmark = reference_frame->features[match.trainIdx]->landmark.lock();
        new_frame->features[match.queryIdx]->landmark = landmark;
        landmark->observations.push_back(new_frame->features[match.queryIdx]);

        // NOTE: only the topology is being updated here; numerical values will be adjusted via bundle adjustment
    }
}

}

Backend::Backend(const Parameters& parameters, const Camera& rgb_camera, const Camera& depth_camera, Map& map)
    : parameters_(parameters), rgb_camera_(rgb_camera), depth_camera_(depth_camera),
        map_(map), matcher_(cv::NORM_HAMMING, true) {}

void Backend::HandleRgbdFrameEvent(const RgbdFrameEvent& frame) {
    auto previous_frame = map_.GetMostRecentKeyframe();
    auto keyframe = map_.CreateKeyframe(frame);

    if (!previous_frame) {
        // This is the first frame; let's just create landmarks for the features we have
        keyframe->pinned = true;
        map_.CreateLandmarksForUnmappedFeatures(*frame.info.rgb, keyframe);
        return;
    }

    // Ensure we have consistent ordering in time
    assert(previous_frame->timestamp < keyframe->timestamp);
    auto previous_frame_matches = MatchFeatures(previous_frame->descriptions, keyframe->descriptions);

    if (previous_frame_matches.size() > parameters_.min_feature_matches) {
        MatchFeaturesToLandmarks(previous_frame, keyframe, previous_frame_matches);
        map_.CreateLandmarksForUnmappedFeatures(*frame.info.rgb, keyframe);

        Keyframes keyframes;
        Landmarks landmarks;

        ExtractLocalGraph(keyframe, keyframes, landmarks);

        Poses poses;
        Locations locations;

        OptimizePosesAndLocations(keyframes, landmarks, poses, locations);
        UpdatePosesAndLocations(keyframes, landmarks, poses, locations);

        // TODO: Perform a loop-closure test; potentially trigger a global BA
    } else {
        // perform a database search and position the frame against the best hits from
        // the search

        // match against best frame found

        // Compute a local, bundle adjustment using connectivity information
    }
}

std::vector<cv::DMatch> Backend::MatchFeatures(const cv::Mat& reference, const cv::Mat& query) {
    std::vector<cv::DMatch> matches;
    matcher_.match(reference, query, matches);

    // trim matches based on parameters.max_match_distance
    std::sort(matches.begin(), matches.end());

    auto iter = matches.begin();
    for (; iter != matches.end(); ++iter) {
        if (iter->distance > parameters_.max_match_distance) {
            break;
        }
    }

    matches.erase(iter, matches.end());
    return matches;
}

void Backend::ExtractLocalGraph(const KeyframePointer& keyframe, Keyframes& keyframes,
                                Landmarks& landmarks) {
    using std::begin, std::end;

    // This is used as max heap of keyframe timestamps to drive the BFS 
    std::vector<Timestamp> timestamp_heap;

    std::map<Timestamp, KeyframePointer> keyframes_seen;
    std::unordered_map<LandmarkId, LandmarkPointer> landmarks_seen;

    keyframes_seen[keyframe->timestamp] = keyframe;
    timestamp_heap.push_back(keyframe->timestamp);

    keyframes.clear();
    landmarks.clear();

    while (keyframes.size() < parameters_.max_keyframes_in_local_graph && !timestamp_heap.empty()) {
        // pop the highest key frame
        std::pop_heap(begin(timestamp_heap), end(timestamp_heap));
        auto frame = map_.GetKeyframe(timestamp_heap.back());
        timestamp_heap.pop_back();

        // add keyframe to result sub graph in case it is not a pinned one
        if (!frame->pinned) {
            keyframes.push_back(frame);
        }

        // make sure we won't visit this frame again
        keyframes_seen[frame->timestamp] = frame;

        // add all landmarks that are observed by the keyframe and that we have not seen yet
        for (const auto& feature: frame->features) {
            if (auto landmark = feature->landmark.lock()) {
                if (landmarks_seen.find(landmark->id) != landmarks_seen.end()) {
                    continue;
                }

                landmarks.push_back(landmark);
                landmarks_seen[landmark->id] = landmark;

                for (const auto& observation: landmark->observations) {
                    if (auto feature = observation.lock()) {
                        if (auto keyframe = feature->keyframe.lock()) {
                            if (keyframes_seen.find(keyframe->timestamp) != keyframes_seen.end()) {
                                continue;
                            }

                            keyframes_seen[keyframe->timestamp] = keyframe;
                            timestamp_heap.push_back(keyframe->timestamp);
                            std::push_heap(begin(timestamp_heap), end(timestamp_heap));
                        }
                    }
                }
            }
        }  
    } 
}

void Backend::OptimizePosesAndLocations(const Keyframes& keyframes, const Landmarks& landmarks,
                                        Poses& poses, Locations& locations) {
    g2o::SparseOptimizer optimizer;

    typedef g2o::BlockSolver<g2o::BlockSolverTraits<6, 3>> BlockSolverType;
    typedef g2o::LinearSolverCSparse<BlockSolverType::PoseMatrixType> LinearSolverType;

    auto algorithm = new g2o::OptimizationAlgorithmLevenberg(
        g2o::make_unique<BlockSolverType>(g2o::make_unique<LinearSolverType>()));

    optimizer.setAlgorithm(algorithm);
    optimizer.setVerbose(false);

    std::map<Timestamp, size_t> keyframe_indices;
    std::map<LandmarkId, size_t> landmark_indices;

    for (size_t index = 0; index < keyframes.size(); ++index) {
        const auto& keyframe = keyframes[index];

        keyframe_indices[keyframe->timestamp] = index;

        auto vertex = new g2o::VertexSE3();
        vertex->setId(index);
        auto rotation = keyframe->pose.unit_quaternion();
        auto translation = keyframe->pose.translation();
        g2o::Isometry3 pose = Eigen::Translation3d(translation) * Eigen::AngleAxisd(rotation);
        vertex->setEstimate(pose);
        vertex->setFixed(keyframe->pinned);
        optimizer.addVertex(vertex);
    }

    auto landmark_vertex_offset = keyframes.size();

    for (size_t index = 0; index < landmarks.size(); ++index) {
        const auto& landmark = landmarks[index];

        landmark_indices[landmark->id] = index;

        auto vertex = new g2o::VertexPointXYZ();
        vertex->setId(landmark_vertex_offset + index);
        vertex->setEstimate(landmark->location);
        optimizer.addVertex(vertex);
    }

    for (size_t keyframe_index = 0; keyframe_index < keyframes.size(); ++keyframe_index) {
        const auto& keyframe = keyframes[keyframe_index];

        for (const auto& feature: keyframe->features) {
            if (auto landmark = feature->landmark.lock()) {
                auto landmark_iter = landmark_indices.find(landmark->id);

                // Skip landmarks that don't contribute to the sub graph at hand
                if (landmark_iter == landmark_indices.end()) {
                    // should only happen for keyframes that anchor the optimization problem
                    assert(keyframe->pinned);
                    continue;
                }

                auto landmark_index = landmark_iter->second;

                auto edge = new g2o::EdgeSE3PointXYZDepth();
                edge->setVertex( 0, optimizer.vertex(keyframe_index));
                edge->setVertex( 1, optimizer.vertex(landmark_index + landmark_vertex_offset));
                // TODO: At some point, we need to use robot coordinates instead of camera coordinates
                Point3d measurement = rgb_camera_.PixelToCamera(feature->keypoint.pt, feature->depth);
                edge->setMeasurement(measurement);

                // TODO: Apply error model for depth measurement
                edge->setInformation(Eigen::Matrix3d::Identity());

                // What is this?
                edge->setParameterId(0, 0);
                edge->setRobustKernel(new g2o::RobustKernelHuber());
                optimizer.addEdge(edge);
            }
        }
    }

    // Perform the actual optimization
    optimizer.setVerbose(true); // while debugging
    optimizer.initializeOptimization();
    optimizer.optimize(parameters_.local_optimization_iterations);

    // retrieve results
    poses.clear();

    for (size_t index = 0; index < keyframes.size(); ++index) {
        auto vertex = dynamic_cast<g2o::VertexSE3 *>(optimizer.vertex(index));
        auto isometry = vertex->estimate();
        auto rotation = isometry.rotation();
        auto translation = isometry.translation();
        SE3d pose(rotation, translation);
        poses.push_back(pose);
    }

    locations.clear();

    for (size_t index = 0; index < landmarks.size(); ++index) {
        auto vertex = dynamic_cast<g2o::VertexPointXYZ *>(optimizer.vertex(index + landmark_vertex_offset));
        locations.push_back(vertex->estimate());

        // TODO: What criteria should we apply to mark bad features? 
        double error = 0.0;
        for (auto edge: vertex->edges()) {
            auto se3_point_xyz_depth = dynamic_cast<g2o::EdgeSE3PointXYZDepth*>(edge);
            se3_point_xyz_depth->computeError();
            auto chi2 = se3_point_xyz_depth->chi2();
        }
    }
}

void Backend::UpdatePosesAndLocations(const Keyframes& keyframes, const Landmarks& landmarks,
                                      const Poses& poses, const Locations& locations) {
    assert(keyframes.size() == poses.size());
    assert(landmarks.size() == locations.size());

    // TODO: Check for concurrency issues once we go multi-threaded
    for (size_t index = 0; index < keyframes.size(); ++index) {
        auto& keyframe = keyframes[index];

        if (!keyframe->pinned) {
            keyframe->pose = poses[index];
        }
    }

    for (size_t index = 0; index < landmarks.size(); ++index) {
        auto& landmark = landmarks[index];
        landmark->location = locations[index];

        // Update normal
        Vector3d normal_sum;
        for (const auto& observation: landmark->observations) {
            if (auto feature = observation.lock()) {
                if (auto keyframe = feature->keyframe.lock()) {
                    normal_sum -= (keyframe->pose * landmark->location).normalized();
                }
            }
        }

        landmark->normal = normal_sum.normalized();
    }
}
