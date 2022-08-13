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

#ifndef SLAMMER_IMU_H
#define SLAMMER_IMU_H

#pragma once

#include "slammer/slammer.h"


namespace slammer {

// forward declarations: Events
class AccelerometerEvent;
class GyroscopeEvent;

struct ImuData {
    Vector3d acceleration;
    Vector3d rotation;

    Timestamp time_acceleration;
    Timestamp time_rotation;
};

/// The class processing accelerator and gyroscope measurement generated by an IMU sensor.
class Imu {
public:
    /// determine whether accelerometer or gyroscope trigger processing of the next frame
    enum class Trigger {
        /// accelerometer or gyroscope trigger processing of the next frame
        kTriggerAccelerometer,

        /// accelerometer or gyroscope trigger processing of the next frame
        kTriggerGyroscope,

        /// both events trigger recalculation
        kTriggerBoth
    };

    /// Specify which event should trigger recalculation of the pose
    void set_trigger(Trigger trigger) { trigger_ = trigger; }

    /// Retrieve the processing trigger
    Trigger trigger() const { return trigger_; }

    /// Retrieve the last timestamp for which we estimated a pose
    Timestamp timestamp_pose() const { return timestamp_pose_; }

    /// Retrieve the last estimated pose
    SE3d pose() const { return pose_; }

    // Event handlers
    void HandleAccelerometerEvent(const AccelerometerEvent& event);
    void HandleGyroscopeEvent(const GyroscopeEvent& event);

private:
    /// Process the next frame and create an updated pose estimation
    void ProcessFrame();

    // which event is triggering processing
    Trigger trigger_;

    // timestamp of last estimated pose
    Timestamp timestamp_pose_;

    // last estimated pose
    SE3d pose_;

    // IMU readings
    ImuData imu_data_;
};

} // namespace slammer

#endif //ndef SLAMMER_IMU_H
