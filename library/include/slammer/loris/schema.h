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

#ifndef SLAMMER_LORIS_SCHEMA_H
#define SLAMMER_LORIS_SCHEMA_H

#pragma once

#include "arrow/api.h"

namespace slammer {
namespace loris {

struct D435Tables {
    std::shared_ptr<arrow::Table> aligned_depth;
    std::shared_ptr<arrow::Table> color;
    std::shared_ptr<arrow::Table> d400_accelerometer;
    std::shared_ptr<arrow::Table> d400_gyroscope;
    std::shared_ptr<arrow::Table> depth;
};

extern std::shared_ptr<arrow::Schema> aligned_depth_schema;
extern std::shared_ptr<arrow::Schema> color_schema;
extern std::shared_ptr<arrow::Schema> d400_accelerometer_schema;
extern std::shared_ptr<arrow::Schema> d400_gyroscope_schema;
extern std::shared_ptr<arrow::Schema> depth_schema;

struct T265Tables {
    std::shared_ptr<arrow::Table> fisheye1;
    std::shared_ptr<arrow::Table> fisheye2;
    std::shared_ptr<arrow::Table> t265_accelerometer;
    std::shared_ptr<arrow::Table> t265_gyroscope;
};

extern std::shared_ptr<arrow::Schema> fisheye1_schema;
extern std::shared_ptr<arrow::Schema> fisheye2_schema;
extern std::shared_ptr<arrow::Schema> t265_accelerometer_schema;
extern std::shared_ptr<arrow::Schema> t265_gyroscope_schema;

struct CommonTables {
    std::shared_ptr<arrow::Table> groundtruth;
    std::shared_ptr<arrow::Table> odom;
};

extern std::shared_ptr<arrow::Schema> groundtruth_schema;
extern std::shared_ptr<arrow::Schema> odom_schema;

} // namespace loris
} // namespace slammer

#endif //ndef SLAMMER_LORIS_SCHEMA_H
