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

#include "slammer/pipeline.h"


using namespace slammer;


Pipeline::Pipeline(const RgbdFrontend::Parameters& frontend_parameters,
                   const Backend::Parameters& backend_parameters,
                   Vocabulary&& vocabulary, 
                   Camera&& rgb_camera, Camera&& depth_camera,
                   EventListenerList<ColorImageEvent>& color_source,
                   EventListenerList<DepthImageEvent>& depth_source)
    :   rgb_camera_(std::move(rgb_camera)),
        depth_camera_(std::move(depth_camera)),
        frontend_(frontend_parameters, rgb_camera_, depth_camera_),
        keyframe_index_(std::move(vocabulary)),
        backend_(backend_parameters, rgb_camera_, depth_camera_, map_, keyframe_index_)
{
    using namespace std::placeholders;
    
    frontend_.keyframes.AddHandler(std::bind(&Backend::HandleRgbdFrameEvent, &backend_, _1));
    backend_.keyframe_poses.AddHandler(std::bind(&RgbdFrontend::HandleKeyframePoseEvent, &frontend_, _1));
    
    color_source.AddHandler(std::bind(&RgbdFrontend::HandleColorEvent, &frontend_, _1));
    depth_source.AddHandler(std::bind(&RgbdFrontend::HandleDepthEvent, &frontend_, _1));
}
