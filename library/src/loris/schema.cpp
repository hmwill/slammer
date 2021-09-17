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

#include "slammer/loris/schema.h"

namespace {

std::shared_ptr<arrow::Schema> CreateImageListSchema() {
    return arrow::schema({
        arrow::field("Time", arrow::float64()), 
        arrow::field("Path", arrow::utf8())
    });
}

std::shared_ptr<arrow::Schema> CreateAccelerometerSchema() {
    return arrow::schema({
        arrow::field("Time", arrow::float64()), 
        arrow::field("Ax", arrow::float64()),
        arrow::field("Ay", arrow::float64()),
        arrow::field("Az", arrow::float64())
    });
}

std::shared_ptr<arrow::Schema> CreateGyroscopeSchema() {
    return arrow::schema({
        arrow::field("Time", arrow::float64()), 
        arrow::field("Gx", arrow::float64()),
        arrow::field("Gy", arrow::float64()),
        arrow::field("Gz", arrow::float64())
    });
}

} // namespace

std::shared_ptr<arrow::Schema> slammer::loris::aligned_depth_schema = CreateImageListSchema();
std::shared_ptr<arrow::Schema> slammer::loris::color_schema = CreateImageListSchema();
std::shared_ptr<arrow::Schema> slammer::loris::d400_accelerometer_schema = CreateAccelerometerSchema();
std::shared_ptr<arrow::Schema> slammer::loris::d400_gyroscope_schema = CreateGyroscopeSchema();
std::shared_ptr<arrow::Schema> slammer::loris::depth_schema = CreateImageListSchema();

std::shared_ptr<arrow::Schema> slammer::loris::fisheye1_schema = CreateImageListSchema();
std::shared_ptr<arrow::Schema> slammer::loris::fisheye2_schema = CreateImageListSchema();
std::shared_ptr<arrow::Schema> slammer::loris::t265_accelerometer_schema = CreateAccelerometerSchema();
std::shared_ptr<arrow::Schema> slammer::loris::t265_gyroscope_schema = CreateGyroscopeSchema();

namespace {

std::shared_ptr<arrow::Schema> CreateGroundtruthSchema() {
    return arrow::schema({
        arrow::field("Time", arrow::float64()), 
        arrow::field("px", arrow::float64()),
        arrow::field("py", arrow::float64()),
        arrow::field("pz", arrow::float64()),
        arrow::field("qx", arrow::float64()),
        arrow::field("qy", arrow::float64()),
        arrow::field("qz", arrow::float64()),
        arrow::field("qw", arrow::float64())
    });
}

std::shared_ptr<arrow::Schema> CreateOdomSchema() {
    return arrow::schema({
        arrow::field("Time", arrow::float64()), 
        arrow::field("pose.position.x", arrow::float64()),
        arrow::field("pose.position.y", arrow::float64()),
        arrow::field("pose.position.z", arrow::float64()),
        arrow::field("pose.orientation.x", arrow::float64()),
        arrow::field("pose.orientation.y", arrow::float64()),
        arrow::field("pose.orientation.z", arrow::float64()),
        arrow::field("pose.orientation.w", arrow::float64()),
        arrow::field("twist.linear.x", arrow::float64()),
        arrow::field("twist.linear.y", arrow::float64()),
        arrow::field("twist.linear.z", arrow::float64()),
        arrow::field("twist.angular.x", arrow::float64()),
        arrow::field("twist.angular.y", arrow::float64()),
        arrow::field("twist.angular.z", arrow::float64())
    });
}

} // namespace

std::shared_ptr<arrow::Schema> slammer::loris::groundtruth_schema = CreateGroundtruthSchema();
std::shared_ptr<arrow::Schema> slammer::loris::odom_schema = CreateOdomSchema();
