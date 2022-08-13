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

#ifndef SLAMMER_LORIS_DRIVER_H
#define SLAMMER_LORIS_DRIVER_H

#pragma once

#include "slammer/slammer.h"
#include "slammer/events.h"

#include <optional>

#include "rapidcsv.h"

namespace slammer {
namespace loris {

class Driver;
class AbstractEventSource;

struct HeapEntry {
    Timestamp timestamp;
    std::unique_ptr<AbstractEventSource> event_source;
};

/// Ground truth values
struct GroundtruthEvent: public Event {
    Point3d position;
    Quaterniond orientation;
};

class Driver {
public:
    Driver(const std::string& path);
    ~Driver();

    Result<size_t> Run(const std::optional<Timediff> max_duration = {}, 
                       const std::optional<size_t> max_num_events = {});

    const std::string& path() const { return path_; }

    EventListenerList<AccelerometerEvent> d400_accelerometer;
    EventListenerList<GyroscopeEvent> d400_gyroscope;
    EventListenerList<DepthImageEvent> aligned_depth;
    EventListenerList<DepthImageEvent> depth;
    EventListenerList<ColorImageEvent> color;

    EventListenerList<AccelerometerEvent> t265_accelerometer;
    EventListenerList<GyroscopeEvent> t265_gyroscope;
    EventListenerList<FisheyeImageEvent> fisheye1;
    EventListenerList<FisheyeImageEvent> fisheye2;

    EventListenerList<GroundtruthEvent> groundtruth;

private:
    std::optional<Error> AddEventSource(std::unique_ptr<AbstractEventSource>&& event_source);

    std::string path_;
    std::vector<HeapEntry> timestamp_heap_;
};

} // namespace loris
} // namespace slammer

#endif //ndef SLAMMER_LORIS_DRIVER_H
