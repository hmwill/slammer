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

Backend::Backend(const Parameters& parameters, Map& map)
    : parameters_(parameters), map_(map), matcher_(cv::NORM_HAMMING, true) {}

void Backend::HandleRgbdFrameEvent(const RgbdFrameEvent& frame) {
    auto previous_frame = map_.GetMostRecentKeyframe();
    auto keyframe = map_.CreateKeyframe(frame);

    if (!previous_frame) {
        // This is the first frame; let's just create landmarks for the features we have
        map_.CreateLandmarksForUnmappedFeatures(*frame.info.rgb, keyframe);
        return;
    }

    auto previous_frame_matches = MatchFeatures(previous_frame->descriptions, keyframe->descriptions);

    if (previous_frame_matches.size() > parameters_.min_feature_matches) {
        MatchFeaturesToLandmarks(previous_frame, keyframe, previous_frame_matches);
        map_.CreateLandmarksForUnmappedFeatures(*frame.info.rgb, keyframe);

        // TODO: Compute a local, windowed bundle adjustment
        // Question: what sub-graph should we extract?
        // How to create the BA optimization problem? How to handle boundary conditions
        // What solver to use and how to invoke it?
        // How to update poses and landmarks after the optimization has completed?

        // TODO: Perform a loop-closure test; potentially trigger a global BA
    } else {
        // perform a database search and position the frame against the best hits from
        // the search

        // match against best frame found

        // Compute a local, bundle adjustment using connectivity information
    }
}

std::vector<cv::DMatch> Backend::MatchFeatures(const cv::Mat& descriptions1, const cv::Mat& descriptions2) {
    std::vector<cv::DMatch> matches;
    matcher_.match(descriptions1, descriptions2, matches);

    // trim matches based on parameters.max_match_distance
    std::sort(matches.begin(), matches.end());

    auto iter = matches.begin();
    while (iter != matches.end()) {
        if (iter->distance > parameters_.max_match_distance)
            break;

        ++iter;
    }

    matches.erase(iter, matches.end());
    return matches;
}
