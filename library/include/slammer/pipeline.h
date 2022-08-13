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

#ifndef SLAMMER_PIPELINE_H
#define SLAMMER_PIPELINE_H

#pragma once

#include "slammer/slammer.h"
#include "slammer/events.h"

#include "slammer/backend.h"
#include "slammer/camera.h"
#include "slammer/frontend.h"
#include "slammer/map.h"
#include "slammer/keyframe_index.h"

namespace slammer {

class Pipeline {
public:
    Pipeline(const FrontendParameters& frontend_parameters,
             const Backend::Parameters& backend_parameters,
             Vocabulary&& vocabulary, Camera&& rgb_camera, StereoDepthCamera&& depth_camera,
             EventListenerList<ColorImageEvent>& color_source,
             EventListenerList<DepthImageEvent>& depth_source
);

    Pipeline(const Pipeline&) = delete;
    Pipeline& operator=(const Pipeline&) = delete;

private:
    Camera rgb_camera_;
    StereoDepthCamera depth_camera_;

    Map map_;

    Frontend frontend_;
    KeyframeIndex keyframe_index_;
    Backend backend_;
};

} // namespace slammer

#endif //ndef SLAMMER_PIPELINE_H
