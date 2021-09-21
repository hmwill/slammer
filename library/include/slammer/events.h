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

#ifndef SLAMMER_EVENTS_H
#define SLAMMER_EVENTS_H

#pragma once

#include "slammer/slammer.h"

#include <functional>
#include <vector>
#include <optional>

#include "Eigen/Dense"
#include "opencv2/opencv.hpp"

namespace slammer {

using Timestamp = double;
using Vector3d = Eigen::Vector3d;
using Image = cv::Mat;
using ImagePointer = std::shared_ptr<Image>;

/// Common base class for different types of events
struct Event {
    Timestamp timestamp;
};

/// Accelerometer information
struct AccelerometerEvent: public Event {
    Vector3d acceleration;
};

/// Rotation readings provided by a gyroscope
struct GyroscopeEvent: public Event {
    Vector3d rotation;
};

/// An image captured by a form of imaging device/camera
struct ImageEvent: public Event {
    ImagePointer image;
};

/// Steady clock beat creatad by a timer
struct TimerEvent: public Event { };


/// A machanism to register handlers that would like to receive notifications for
/// events generated by a given source
template <typename EventType>
class EventListenerList {
public:
    using HandlerType = std::function<void(const EventType&)>;

    /// Register a new event handler with this event source. This will be called by
    /// subscribers to the event source.
    void AddHandler(std::function<void(const EventType&)>&& handler) {
        handlers_.emplace_back(handler);
    }

    /// Pass an event to all subscribed event handlers
    void HandleEvent(const EventType& event) const {
        for (const auto& handler: handlers_) {
            handler(event);
        }
    }

private:
    std::vector<HandlerType> handlers_;
};


} // namespace slammer

#endif //ndef SLAMMER_EVENTS_H