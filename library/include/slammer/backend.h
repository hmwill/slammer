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

namespace slammer {

class Map;

/// \brief This class provides the backend process for Slammer
class Backend {
public:
    /// Configuration parameters for the backend process 
    struct Parameters {
        // maximum (Hamming) difference between features to be considered a match
        float max_match_distance = 12.0f;

        // minimum number of matches against previous frame to work locally
        int min_feature_matches = 50;
    };

    Backend(const Parameters& parameters, Map& map);

    // Disallow copy construction and copy assignment
    Backend(const Backend&) = delete;
    Backend& operator=(const Backend&) = delete;

    /// Handler function that is called by the frontend when a new frame is available
    void HandleRgbdFrameEvent(const RgbdFrameEvent& frame);

private:
    std::vector<cv::DMatch> MatchFeatures(const cv::Mat& descriptions1, const cv::Mat& descriptions2);

private:
    /// Configuration parameters
    Parameters parameters_;

    /// The sparse map we are populating
    Map& map_;

    /// Feature matcher
    cv::BFMatcher matcher_;
};

} // namespace slammer

#endif //ndef SLAMMER_BACKEND_H