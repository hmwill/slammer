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

#include "slammer/backend.h"

#include "slammer/map.h"
#include "slammer/math.h"
#include "slammer/pnp.h"

#include "slammer/pose_graph_optimizer.h"
#include "slammer/poses_locations_optimizer.h"


using namespace slammer;

namespace {

// Link all features in the incoming new frame that have been matched against a given reference
// frame to the landmarks already associated with their corresponding feature.
void MatchFeaturesToLandmarks(const KeyframePointer& reference_frame, const KeyframePointer& new_frame,
                              const std::vector<Match>& matches) {
    for (const auto& match: matches) {
        auto landmark = reference_frame->features[match.target_index]->landmark.lock();
        new_frame->features[match.query_index]->landmark = landmark;
        landmark->observations.push_back(new_frame->features[match.query_index]);
    }
}

} // namespace

Backend::Backend(const Parameters& parameters, const Camera& rgb_camera, const StereoDepthCamera& depth_camera, 
                 Map& map, KeyframeIndex& keyframe_index)
    : parameters_(parameters), rgb_camera_(rgb_camera), depth_camera_(depth_camera),
        map_(map), keyframe_index_(keyframe_index) {}

void Backend::HandleKeyframeEvent(const KeyframeEvent& frame) {
    auto previous_frame = map_.GetMostRecentKeyframe();
    auto keyframe = map_.CreateKeyframe(frame);
    auto original_pose = keyframe->pose;

    if (!previous_frame) {
        // This is the first frame; let's just create landmarks for the features we have
        keyframe->pinned = true;
        map_.CreateLandmarksForUnmappedFeatures(frame.rgb_camera, keyframe);
        return;
    }

    // Ensure we have consistent ordering in time
    assert(previous_frame->timestamp < keyframe->timestamp);
    auto previous_frame_matches = MatchFeatures(previous_frame->descriptions, keyframe->descriptions);

    if (previous_frame_matches.size() > parameters_.min_feature_matches) {
        ExtendGraph(keyframe, previous_frame, previous_frame_matches);

        KeyframePointer loop_keyframe;
        SE3d relative_motion;
        LandmarkMapping landmark_mapping;

        if (DetermineLoopClosure(keyframe, loop_keyframe, relative_motion, landmark_mapping)) {

            Keyframes keyframes;
            Landmarks landmarks;

            Keyframes seeds { keyframe, loop_keyframe };

            ExtractLocalGraph(seeds, keyframes, landmarks, std::numeric_limits<size_t>::max());
            Poses poses = OptimizeLoopPoses(keyframes, keyframe, loop_keyframe, relative_motion);

            Landmarks filtered_landmarks;

            std::copy_if(landmarks.begin(), landmarks.end(), std::back_inserter(filtered_landmarks),
                [&landmark_mapping](const auto& landmark) { 
                    return landmark_mapping.find(landmark->id) == landmark_mapping.end();
                }
            );

            Locations locations = EstimateLocations(keyframes, filtered_landmarks, poses);
            OptimizePosesAndLocations(keyframes, filtered_landmarks, poses, locations, landmark_mapping, true);

            std::for_each(landmark_mapping.begin(), landmark_mapping.end(),
                [&](const auto& mapping){ map_.MergeLandmarks(mapping.second, mapping.first); });

            UpdatePosesAndLocations(keyframes, filtered_landmarks, poses, locations);
        }
    } else {
        std::vector<KeyframeIndex::Result> loop_candidates;
        keyframe_index_.Search(keyframe, loop_candidates, parameters_.max_loop_candidiates);

        KeyframePointer previous_frame;
        std::vector<Match> previous_frame_matches;
        double match_score = 0.0;

        for (const auto& result: loop_candidates) {
            auto matches = MatchFeatures(keyframe->descriptions, result.keyframe->descriptions);

            if (matches.size() < parameters_.min_loop_feature_matches) {
                continue;
            }

            std::vector<Point3d> keyframe_points, candidate_points;
            PerspectiveAndPoint3d::PointPairs point_pairs;

            for (const auto& match: matches) {
                keyframe_points.push_back(keyframe->features[match.target_index]->coords);
                candidate_points.push_back(keyframe->features[match.query_index]->coords);
                PerspectiveAndPoint3d::PointPair point_pair { 
                    keyframe->features[match.target_index]->coords,
                    keyframe->features[match.query_index]->coords
                };
                point_pairs.push_back(point_pair);
            }

            SE3d relative_motion;
            std::vector<uchar> mask;

            PerspectiveAndPoint3d instance { depth_camera_, depth_camera_, point_pairs };

            Result<double> pnp_result = 
                instance.Ransac(relative_motion, mask, keyframe_points,
                                parameters_.pnp_parameters.sample_size, parameters_.pnp_parameters.max_iterations, 
                                parameters_.pnp_parameters.lambda, parameters_.pnp_parameters.lambda, 
                                random_engine_);

            assert(pnp_result.ok());

            size_t num_inliers = std::count_if(mask.begin(), mask.end(), [](auto flag) { return flag != 0; });

            if (num_inliers < parameters_.min_inlier_matches) {
                continue;
            }    

            double score = result.score * num_inliers;

            if (score > match_score) {
                match_score = score;
                previous_frame_matches = matches;
                previous_frame = result.keyframe;
            }
        }

        if (previous_frame) {
            MatchFeaturesToLandmarks(previous_frame, keyframe, previous_frame_matches);
            map_.CreateLandmarksForUnmappedFeatures(rgb_camera_, keyframe);
            map_.CreateCovisibilityEdges(keyframe);
            keyframe->neighbors.insert(previous_frame);

            Keyframes keyframes;
            Landmarks landmarks;
            Keyframes seeds { keyframe };

            ExtractLocalGraph(seeds, keyframes, landmarks, parameters_.max_keyframes_in_local_graph);

            Poses poses;
            Locations locations;

            OptimizePosesAndLocations(keyframes, landmarks, poses, locations);
            UpdatePosesAndLocations(keyframes, landmarks, poses, locations);
        } else {
            map_.CreateLandmarksForUnmappedFeatures(frame.rgb_camera, keyframe);
        }
    }

    // Notify listeners of an update keyframe pose
    KeyframePoseEvent event;
    event.timestamp = keyframe->timestamp;
    event.keyframe = keyframe;
    event.previous_pose = original_pose;

    keyframe_poses.HandleEvent(event);
}

void Backend::ExtendGraph(const KeyframePointer& keyframe, const KeyframePointer& reference_frame,
                          const FeatureMatches& matches) {
    MatchFeaturesToLandmarks(reference_frame, keyframe, matches);
    map_.CreateLandmarksForUnmappedFeatures(rgb_camera_, keyframe);
    map_.CreateCovisibilityEdges(keyframe);
    keyframe->neighbors.insert(reference_frame);

    Keyframes keyframes;
    Landmarks landmarks;
    Keyframes seeds { keyframe };

    ExtractLocalGraph(seeds, keyframes, landmarks, parameters_.max_keyframes_in_local_graph);

    Poses poses;
    Locations locations;

    OptimizePosesAndLocations(keyframes, landmarks, poses, locations);
    UpdatePosesAndLocations(keyframes, landmarks, poses, locations);
}

bool Backend::DetermineLoopClosure(const KeyframePointer& keyframe, KeyframePointer& loop_keyframe,
                                   SE3d& relative_motion, LandmarkMapping& landmark_mapping) {
    std::vector<KeyframeIndex::Result> loop_candidates;
    keyframe_index_.Search(keyframe, loop_candidates, parameters_.max_loop_candidiates);

    for (const auto& result: loop_candidates) {
        auto matches = MatchFeatures(keyframe->descriptions, result.keyframe->descriptions);

        if (matches.size() < parameters_.min_loop_feature_matches) {
            continue;
        }

        std::vector<Point3d> keyframe_points, candidate_points;
        PerspectiveAndPoint3d::PointPairs point_pairs;

        for (const auto& match: matches) {
            keyframe_points.push_back(keyframe->features[match.target_index]->coords);
            candidate_points.push_back(keyframe->features[match.query_index]->coords);
            PerspectiveAndPoint3d::PointPair point_pair { 
                keyframe->features[match.target_index]->coords,
                keyframe->features[match.query_index]->coords
            };
            point_pairs.push_back(point_pair);
        }

        SE3d relative_motion;
        std::vector<uchar> mask;

        PerspectiveAndPoint3d instance { depth_camera_, depth_camera_, point_pairs };

        Result<double> pnp_result = 
            instance.Ransac(relative_motion, mask, keyframe_points,
                            parameters_.pnp_parameters.sample_size, parameters_.pnp_parameters.max_iterations, 
                            parameters_.pnp_parameters.lambda, parameters_.pnp_parameters.lambda, 
                            random_engine_);

        assert(pnp_result.ok());

        size_t num_inliers = std::count_if(mask.begin(), mask.end(), [](auto flag) { return flag != 0; });

        if (num_inliers < parameters_.min_inlier_matches) {
            continue;
        }    

        // Characterize the correction amount for the proposed loop closure
        auto loop_correction = relative_motion.inverse() * (keyframe->pose * result.keyframe->pose.inverse());
        auto loop_correction_rotation = loop_correction.unit_quaternion();
        auto loop_correction_angle = 
            2.0 * atan2(loop_correction_rotation.vec().norm(), fabs(loop_correction_rotation.w()));
        auto loop_correction_distance = loop_correction.translation().norm();

        if (loop_correction_distance > parameters_.max_loop_correction_distance ||
            loop_correction_angle > parameters_.max_loop_correction_angle) {
            // Skipping this candidate for loop closure; amount of correction is suspiciously high
            // Question: Is there truly a meanigful constant upper limit?
            continue;
        }

        KeyframePointer loop_keyframe = result.keyframe;

        // TODO: #1 Change the logic to return best candidate, not first valid candidate
        // https://github.com/hmwill/slammer/issues/1
        for (const auto& match: matches) {
            auto source_landmark = keyframe->features[match.target_index]->landmark.lock();
            auto target_landmark = loop_keyframe->features[match.query_index]->landmark.lock();

            landmark_mapping[source_landmark->id] = target_landmark->id;
        }

        return true;
    }

    return false;
}


std::vector<Match> Backend::MatchFeatures(const Descriptors& reference, const Descriptors& query) {
    return ComputeMatches(reference, query, parameters_.max_match_distance);
}

void Backend::ExtractLocalGraph(const Keyframes& seeds, Keyframes& keyframes,
                                Landmarks& landmarks, size_t subgraph_limit) {
    using std::begin, std::end;

    // This is used as max heap of keyframe timestamps to drive the BFS 
    std::vector<Timestamp> timestamp_heap;

    std::map<Timestamp, KeyframePointer> keyframes_seen;
    std::unordered_map<LandmarkId, LandmarkPointer> landmarks_seen;

    for (const auto& keyframe: seeds) {
        keyframes_seen[keyframe->timestamp] = keyframe;
        timestamp_heap.push_back(keyframe->timestamp);
    }

    keyframes.clear();
    landmarks.clear();

    while (keyframes.size() < subgraph_limit && !timestamp_heap.empty()) {
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

Backend::Poses 
Backend::OptimizeLoopPoses(const Keyframes& keyframes, const KeyframePointer& from, 
                           const KeyframePointer& to, SE3d relative_motion) {

    PoseGraphOptimizer::Keyframes input;

    for (const auto& frame: keyframes) {
        if (frame != from) {
            input.push_back(from.get());
        }
    }

    PoseGraphOptimizer optimizer { input };
    PoseGraphOptimizer::Poses poses { keyframes.size() };

    PoseGraphOptimizer::Parameters parameters {
        20,
        0.1
    };

    PoseGraphOptimizer::Constraints constraints {
        PoseGraphOptimizer::Constraint { from.get(), to.get(), to->pose.inverse() * from->pose}
    };

    Result<double> result = optimizer.Optimize(poses, false, constraints, parameters);

    return poses;
}

Backend::Locations 
Backend::EstimateLocations(const Keyframes& keyframes, const Landmarks& landmarks,
                           const Poses& poses) {
    Locations locations;
    KeyframeSet keyframe_set(keyframes.begin(), keyframes.end());

    for (const auto& landmark: landmarks) {
        auto estimate = landmark->location;

        for (const auto& observation: landmark->observations) {
            auto feature = observation.lock();
            auto keyframe = feature->keyframe.lock();

            if (keyframe_set.find(keyframe) == keyframe_set.end()) {
                continue;
            }

            estimate = keyframe->pose.inverse() * feature->coords;

            break;
        }

        locations.push_back(estimate);
    }

    return locations;
}

void Backend::OptimizePosesAndLocations(const Keyframes& keyframes, const Landmarks& landmarks,
                                        Poses& poses, Locations& locations, const LandmarkMapping& mapping,
                                        bool inout) {
    assert(!inout || poses.size() == keyframes.size());
    assert(!inout || locations.size() == landmarks.size());


    PosesLocationsOptimizer optimizer(depth_camera_, keyframes, landmarks, mapping);
    PosesLocationsOptimizer::Parameters parameters {
        this->parameters_.local_optimization_iterations,
        this->parameters_.local_optimization_lambda
    };

    Result<double> result = optimizer.Optimize(poses, locations, inout, parameters);

    // TODO: Are we doing anything with the result value?
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
