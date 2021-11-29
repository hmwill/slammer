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

#ifndef SLAMMER_BACKEND_H
#define SLAMMER_BACKEND_H

#pragma once

#include "slammer/slammer.h"
#include "slammer/frontend.h"
#include "slammer/map.h"

namespace slammer {

/// \brief This class provides the backend process for Slammer
class Backend {
public:
    using Keyframes = std::vector<KeyframePointer>;
    using Landmarks = std::vector<LandmarkPointer>;
    using Poses = std::vector<SE3d>;
    using Locations = std::vector<Point3d>;

    /// Configuration parameters for the backend process 
    struct Parameters {
        // maximum (Hamming) difference between features to be considered a match
        float max_match_distance = 12.0f;

        // minimum number of matches against previous frame to work locally
        int min_feature_matches = 50;

        // upper limit on the number of keyframes included in local optimization
        int max_keyframes_in_local_graph = 15;

        // number of solver iterations for local optimization
        int local_optimization_iterations = 10;
    };

    Backend(const Parameters& parameters, const Camera& rgb_camera, const Camera& depth_camera, Map& map);

    // Disallow copy construction and copy assignment
    Backend(const Backend&) = delete;
    Backend& operator=(const Backend&) = delete;

    /// Handler function that is called by the frontend when a new frame is available
    void HandleRgbdFrameEvent(const RgbdFrameEvent& frame);

private:
    /// Match features from a new frame against features in a given reference frame.
    ///
    /// \param reference    Feature descriptors associated with the referene frames
    /// \param query        Feature descriptors associated with the query frame
    std::vector<cv::DMatch> MatchFeatures(const cv::Mat& reference, const cv::Mat& query);

    /// Starting from the specified keyframe, extract the sub-graph of keyfraems and landmarks
    /// to use for a local bundle adjustment.
    ///
    /// \param keyframe     the keyframe that anchors the subgraph to be extracted
    /// \param keyframes    the keyframes to be included in the subgraph
    /// \param landmarks    the landmarks to be included in the subgraph
    void ExtractLocalGraph(const KeyframePointer& keyframe, Keyframes& keyframes,
                           Landmarks& landmarks);

    /// Optimize poses and locations within the subgraph induced by the given keyframes and landmarks.
    /// Rather than updating pose and location information in place, we collect them into new
    /// data structures that are aligned based on index.
    ///
    /// \param keyframes    the keyframes included in the subgraph
    /// \param landmarks    the landmarks included in the subgraph
    /// \param poses        the new keyframe poses calculated as result of the optimization process
    /// \param locations    the new landmark locations calculated as result of the optimization process
    void OptimizePosesAndLocations(const Keyframes& keyframes, const Landmarks& landmarks,
                                   Poses& poses, Locations& locations);

    /// Incorporate updated keyframe pose and landmark location information into the map.
    ///
    /// \param keyframes    the keyframes included in the subgraph
    /// \param landmarks    the landmarks included in the subgraph
    /// \param poses        the new keyframe poses to incorporate
    /// \param locations    the new landmark locations to incorporate
    void UpdatePosesAndLocations(const Keyframes& keyframes, const Landmarks& landmarks,
                                 const Poses& poses, const Locations& locations);

private:
    /// Configuration parameters
    Parameters parameters_;

    /// Parameters describing the RGB camera
    Camera rgb_camera_;

    /// Parameters describing the depth camera
    Camera depth_camera_;
    
    /// The sparse map we are populating
    Map& map_;

    /// Feature matcher
    cv::BFMatcher matcher_;

};

} // namespace slammer

#endif //ndef SLAMMER_BACKEND_H