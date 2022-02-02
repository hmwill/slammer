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

#ifndef SLAMMER_MAP_H
#define SLAMMER_MAP_H

#pragma once

#include "slammer/slammer.h"
#include "slammer/orb.h"

#include "absl/container/btree_map.h"
#include "absl/container/btree_set.h"

namespace slammer {

// from camera.h
class Camera;

// from frontend.h
struct RgbdFrameEvent;

// in this file
struct Feature;
struct Landmark;
struct Keyframe;

using KeyframePointer = std::shared_ptr<Keyframe>;
using LandmarkPointer = std::shared_ptr<Landmark>;
using FeaturePointer = std::shared_ptr<Feature>;

using KeyframeSet = std::unordered_set<KeyframePointer>;

class ImageDescriptor;

/// Keyframes are 
struct Keyframe: std::enable_shared_from_this<Keyframe> {
    // the timestamp serves as identifer (for a given camera, that is)
    Timestamp timestamp;

    // Estimated pose of the camera
    SE3d pose;

    // tracked features within the keyframe
    std::vector<FeaturePointer> features;

    // associated descriptors (should those live within the individual features?)
    Descriptors descriptions;

    // Keyframe pose is pinned in space; to be excluded from optimization
    bool pinned;

    // Image descriptor; will be filled in by KeyframeIndex
    std::unique_ptr<ImageDescriptor> descriptor;

    // Covisible frames (should those be ordered by time)
    KeyframeSet covisible;

    // Pose graph edges; always go backwards in time
    KeyframeSet neighbors;

    // TODO: Attached image/point cloud?
};

struct Feature: std::enable_shared_from_this<Feature> {
    // keypoint specification
    orb::KeyPoint keypoint;

    // depth value as measured via depth sensor
    float depth;

    // the keyframe where this observation had been seen
    std::weak_ptr<Keyframe> keyframe;

    // global landmark this feature has been mapped to
    std::weak_ptr<Landmark> landmark;
};

using LandmarkId = std::uint64_t;

struct Landmark: std::enable_shared_from_this<Landmark> {
    // Unique identifier for the landmark
    LandmarkId id;

    // estimated location
    Point3d location;

    // uncertainty around location; orthogonal vectors describing Gaussian
    Matrix3d variances;

    // front-facing normal; estimated from observations
    Vector3d normal;

    // references to the observations within individual key frames
    std::vector<std::weak_ptr<Feature>> observations;
};

/// \brief This class provides an environment map for Slammer
///
/// The map aggegates a consistent view on the environment. Two important
/// components are the position of previously observed landmarks
/// and a 3D occupancy map and model of the environment.
///
/// Currently, the map is limited to a single session. In a later stage,
/// we will allow for handling a kidnapping problem (incl. robot restart),
/// where a map in process of creation can be reconciled against one or
/// more previously generated ones.
class Map {
public:
    Map();
    ~Map();
    
    // Disallow copy construction and copy assignment
    Map(const Map&) = delete;
    Map& operator=(const Map&) = delete;

    /// Create a new keyframe entry based on the event data provided by the frontend
    KeyframePointer CreateKeyframe(const RgbdFrameEvent& event);

    /// Locate a keyframe based on a timestamp
    KeyframePointer GetKeyframe(Timestamp timestamp) const;

    /// Locate the most recently added keyframe
    KeyframePointer GetMostRecentKeyframe() const;

    /// Create landmarks for all unmapped featuress in a keyframe
    void CreateLandmarksForUnmappedFeatures(const Camera& camera, const KeyframePointer& keyframe);

    /// Create a new landmark and register it with the map; the landmark will also
    /// be associated with the initial, defining feature
    LandmarkId CreateLandmark(const Camera& camera, const FeaturePointer& feature);

    /// Merge two landmarks into a single one
    ///
    /// \param primary this landmark will remain after the merge operation
    /// \param secondary this landmark will be removed as part of the merge operation
    void MergeLandmarks(LandmarkId primary, LandmarkId secondary);

    /// Access a given landmark by id
    LandmarkPointer GetLandmark(LandmarkId id) const;

    /// Retrieve all keyframes with covisibility
    /// \param keyframe the keyframe for that all other keyframes observing a shared landmark should
    ///                 be determined
    /// \return A set of all frames sharing at least one common observed landmark. The `keyframe` itself is
    ///         excluded.
    KeyframeSet GetCovisibleKeyframes(const KeyframePointer& keyframe) const;

    /// Create covisibility edges amongst keyframes
    void CreateCovisibilityEdges(const KeyframePointer& keyframe);

    /// Is the map empty?
    bool has_keyframes() const { return !keyframes_.empty(); }

private:
    // Identifier to use for the next landmark to create
    LandmarkId next_landmark_id_;

    // We maintain key frames as a collected ordered by timestamp, which allows us
    // to reconstruct temporal adjacencies between key frames.
    absl::btree_map<Timestamp, std::shared_ptr<Keyframe>> keyframes_;

    // We maintain landmarks as unordered collection. In the future, organizing 
    // landmarks using a spatial data structure may be a better choice. 
    std::unordered_map<LandmarkId, std::shared_ptr<Landmark>> landmarks_;
};

} // namespace slammer

#endif //ndef SLAMMER_MAP_H