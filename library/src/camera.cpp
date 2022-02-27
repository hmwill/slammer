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

#include "slammer/camera.h"

using namespace slammer;


Camera::Camera(float width, float height,
               float fx, float fy, float cx, float cy,
               SE3d camera_to_robot) 
    :   width_(width), height_(height), fx_(fx), fy_(fy), cx_(cx), cy_(cy),
        camera_to_robot_(camera_to_robot) {
    robot_to_camera_ = camera_to_robot_.inverse();
}

Point2f Camera::RobotToPixel(const Point3d& coord) const {
    return CameraToPixel(robot_to_camera_ * coord);
}

Point3d Camera::PixelToRobot(const Point2f& coord, double depth) const {
    return camera_to_robot_ * PixelToCamera(coord, depth);
}

Matrix3d Camera::Jacobian(const Point3d& coord) const {
    Matrix3d result;
    result << 
        fx_ / coord.z(),    0.0,                -fx_ * coord.x() / Square(coord.z()),
        0.0,                fy_ / coord.z(),    -fy_ * coord.y() / Square(coord.z()),
        0.0,                0.0,                1.0;

    return result;
}

Matrix3d StereoDepthCamera::Jacobian(const Point3d& coord) const {
    Matrix3d result;
    result << 
        fx_ / coord.z(),    0.0,                -fx_ * coord.x() / Square(coord.z()),
        0.0,                fy_ / coord.z(),    -fy_ * coord.y() / Square(coord.z()),
        0.0,                0.0,                -fx_ * baseline_ / Square(coord.z());

    return result;
}
