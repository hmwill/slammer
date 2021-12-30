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

#include "slammer/map.h"

#include "slammer/frontend.h"
#include "slammer/keyframe_index.h"

using namespace slammer;

Map::Map() {}

Map::~Map() {
    // because we have cyclic std::shared_ptr references, we break those cycles explicitly
    for (const auto& iter: keyframes_) {
        iter.second->covisible.clear();
        iter.second->neighbors.clear();
    }

    keyframes_.clear();
    landmarks_.clear();
}

KeyframePointer Map::CreateKeyframe(const RgbdFrameEvent& event) {
    auto result = std::make_shared<Keyframe>();

    result->timestamp = event.timestamp;
    result->pose = event.pose;
    result->pinned = false;
    
    for (const auto& keypoint: event.keypoints) {
        auto feature = std::make_shared<Feature>();

        feature->keypoint = keypoint;
        feature->depth = //std::numeric_limits<double>::signaling_NaN();
            static_cast<double>(event.frame_data.depth.at<ushort>(keypoint.pt.y, keypoint.pt.x));

        feature->keyframe = result;

        result->features.emplace_back(std::move(feature));
    }

    result->descriptions = event.descriptions;
    keyframes_[result->timestamp] = result;

    return result;
}

KeyframePointer Map::GetKeyframe(Timestamp timestamp) const {
    auto iter = keyframes_.find(timestamp);

    if (iter != keyframes_.end()) {
        return iter->second;
    } else {
        return KeyframePointer {};
    }
}

KeyframePointer Map::GetMostRecentKeyframe() const {
    auto iter = keyframes_.rbegin();

    if (iter != keyframes_.rend()) {
        return iter->second;
    } else {
        return KeyframePointer {};
    }
}

void Map::CreateLandmarksForUnmappedFeatures(const Camera& camera, const KeyframePointer& keyframe) {
    for (auto& feature: keyframe->features) {
        if (feature->landmark.expired()) {
            auto id = CreateLandmark(camera, feature);
        }
    }
}

LandmarkId Map::CreateLandmark(const Camera& camera, const FeaturePointer& feature) {
    auto keyframe = feature->keyframe.lock();
    // TODO: how to get the coordinates
    Point3d coord = camera.PixelToCamera(feature->keypoint.pt, feature->depth);
    Point3d location = keyframe->pose * coord;

    auto landmark = std::make_shared<Landmark>();
    landmark->id = next_landmark_id_++;
    landmark->location = location;
    landmark->variances = Matrix3d::Identity();
    landmark->normal = -coord.normalized();
    landmark->observations.push_back(feature);
    landmarks_[landmark->id] = landmark;

    feature->landmark = landmark;

    return landmark->id;
}

void Map::MergeLandmarks(LandmarkId primary, LandmarkId secondary) {
    auto landmark_from = GetLandmark(secondary);
    auto landmark_to = GetLandmark(primary);

    KeyframeSet keyframes_from, keyframes_to;

    for (const auto& observation: landmark_to->observations) {
        auto feature = observation.lock();
        keyframes_to.insert(feature->keyframe.lock());
    }

    for (const auto& observation: landmark_from->observations) {
        auto feature = observation.lock();
        feature->landmark = landmark_to;
        keyframes_from.insert(feature->keyframe.lock());
        landmark_to->observations.push_back(feature);
    }

    landmark_from->observations.clear();

    for (const auto& keyframe: keyframes_from) {
        for (const auto& neighbor: keyframes_to) {
            if (neighbor != keyframe) {
                keyframe->covisible.insert(neighbor);
            }
        }
    }

    for (const auto& keyframe: keyframes_to) {
        for (const auto& neighbor: keyframes_from) {
            if (neighbor != keyframe) {
                keyframe->covisible.insert(neighbor);
            }
        }
    }

    landmarks_.erase(landmark_from->id);
}

LandmarkPointer Map::GetLandmark(LandmarkId id) const {
    auto iter = landmarks_.find(id);

    if (iter != landmarks_.end()) {
        return iter->second;
    } else {
        return LandmarkPointer {};
    }
}

KeyframeSet Map::GetCovisibleKeyframes(const KeyframePointer& keyframe) const {
    KeyframeSet result;

    for (const auto& feature: keyframe->features) {
        auto landmark = feature->landmark.lock();

        for (const auto& observation: landmark->observations) {
            auto observed_frame = observation.lock()->keyframe.lock();

            if (observed_frame != keyframe) {
                result.insert(observed_frame);
            }
        }
    }

    return result;
}

void Map::CreateCovisibilityEdges(const KeyframePointer& keyframe) {
    auto other_frames = GetCovisibleKeyframes(keyframe);

    for (const auto& frame: other_frames) {
        frame->covisible.insert(keyframe);
        keyframe->covisible.insert(frame);
    }
}
