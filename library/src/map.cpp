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

using namespace slammer;

KeyframePointer Map::CreateKeyframe(const RgbdFrameEvent& event) {
    auto result = std::make_shared<Keyframe>();

    result->timestamp = event.timestamp;
    result->pose = event.pose;
    result->pinned = false;
    
    for (const auto& keypoint: event.keypoints) {
        auto feature = std::make_shared<Feature>();

        feature->keypoint = keypoint;
        feature->depth = std::numeric_limits<double>::signaling_NaN();
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
    assert(false);
}

LandmarkPointer Map::GetLandmark(LandmarkId id) const {
    auto iter = landmarks_.find(id);

    if (iter != landmarks_.end()) {
        return iter->second;
    } else {
        return LandmarkPointer {};
    }
}
