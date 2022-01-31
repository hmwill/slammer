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

#ifndef SLAMMER_FLOW_H
#define SLAMMER_FLOW_H

#pragma once

#include "slammer/slammer.h"

#include "slammer/image.h"

namespace slammer {

/// Perform an optical flow calculation using the Lucas-Kanade algorithm to map key points
/// in the source image to locations in the target image. Internally, this function will create a
/// power-of-two image pyramid for source and target image.
///
/// \param source                   the source image
/// \param target                   the target image
/// \param source_points            key point locations in the source image
/// \param [inout] target_points    key point loctions in the target image. If `!target_points.empty()` 
///                                 upon calling the function, the current values are used to initialize 
///                                 the target point locations. Otherwise, initialization is done using
///                                 source coordinates.
/// \param [out] error              the error value associated with the results, which can be used to exclude bad 
///                                 matches. 
/// \param num_levels               number of levels to generate for the imge pyramid
/// \param omega                    half-size of window width (window is [-omega .. omega])
/// \param threshold                convergence threshold
void ComputeFlow(const boost::gil::gray8c_view_t& source, const boost::gil::gray8c_view_t& target,
                 const std::vector<Point2f>& source_points, std::vector<Point2f>& target_points,
                 std::vector<float>& error, unsigned num_levels = 4, unsigned omega = 7, float threshold = 0.5f);

} // namespace slammer

#endif //ndef SLAMMER_FLOW_H
