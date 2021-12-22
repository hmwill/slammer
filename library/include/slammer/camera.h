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

#ifndef SLAMMER_CAMERA_H
#define SLAMMER_CAMERA_H

#pragma once

#include "slammer/slammer.h"

namespace slammer {

/// Representation of a camera 
///
/// This is a pinhole camera model, and we assume that the camera has already been
/// calibrated such that no additional distortion needs to be acounted for. This 
/// is an addition we may make at a later stage.
class Camera {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    /// Initialize the camera with its parameters values
    ///
    /// \param width number of pixels along first pixel coodinate
    /// \param height number of pixels along second pixel coordinate
    /// \param fx focal length (x-direction) 
    /// \param fy focal length (y-direction)
    /// \param cx x-coordinate of focal center
    /// \param cy y-coordinate of focal center
    /// \param camera_to_robot Transformation to apply to convert 3D coordinates relative to the camera
    ///     to 3D coordinates for the robot
    Camera(float width, float height,
           float fx, float fy, float cx, float cy,
           SE3d camera_to_robot);

    inline Point3d PixelToCamera(const Point2f& coord, double depth = 1.0) const;
    inline Point2f CameraToPixel(const Point3d& coord) const;

    Point2f RobotToPixel(const Point3d& coord) const;
    Point3d PixelToRobot(const Point2f& coord, double depth = 1.0) const;

    SE3d camera_to_robot() const { return camera_to_robot_; }

    float width() const { return width_; }
    float height() const { return height_; }

    float fx() const { return fx_; }
    float fy() const { return fy_; }
    float cx() const { return cx_; }
    float cy() const { return cy_; }

private:
    /// Transformation matrix of camera frame to robot frame
    SE3d camera_to_robot_;

    /// robot to camera transformation
    SE3d robot_to_camera_;

    /// Instrinsics values
    float fx_, fy_, cx_, cy_;

    /// camera image dimension
    float width_, height_;
};

Point2f Camera::CameraToPixel(const Point3d& coord) const {
    return Point2f(coord(0) / coord(2) * fx_ + cx_,
                   coord(1) / coord(2) * fy_ + cy_);
}

Point3d Camera::PixelToCamera(const Point2f& coord, double depth) const {
    return Point3d((coord.x - cx_) * depth / fx_,
                   (coord.y - cy_) * depth / fy_,
                   depth);
}


} // namespace slammer

#endif //ndef SLAMMER_CAMERA_H
