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

#ifndef SLAMMER_SLAMMER_H
#define SLAMMER_SLAMMER_H

#pragma once

#include <algorithm>
#include <array>
#include <bitset>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <chrono>
#include <exception>
#include <future>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <numeric>
#include <random>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <variant>

#include "boost/dynamic_bitset.hpp"

#include "opencv2/opencv.hpp"
#include "sophus/se3.hpp"

using namespace std::string_literals;

namespace slammer {

// Common types related to time and space

/// Representation of time differences
using Timediff = std::chrono::duration<double>;

/// Representation of absolute time (in seconds, relative to standard OS epoch)
using Timestamp = std::chrono::time_point<std::chrono::system_clock, Timediff>;

/// 2-d coordinates used to address pixels in images
using Point2f = cv::Point2f;

/// 3-d vector type
using Vector3d = Eigen::Vector3d;

/// 3-d matrix type
using Matrix3d = Eigen::Matrix3d;

/// Mathematical affine transformation type for SE(3)
using SE3d = Sophus::SE3d;

/// 3-d point coordinate representation
using Point3d = SE3d::Point;

/// Associated 4-by-4 matrix representation of affine transformations
using Transformation = SE3d::Transformation;

/// Quaternion type to represent rotations
using Quaterniond = SE3d::QuaternionType;

/// The `overloaded` operator, see [Example 4](https://en.cppreference.com/w/cpp/utility/variant/visit).
template<class... Ts> struct overloaded : Ts... { using Ts::operator()...; };
// explicit deduction guide (not needed as of C++20)
template<class... Ts> overloaded(Ts...) -> overloaded<Ts...>;

/// Error type used in this library.
///
/// Error values can be constructed from strings and std::exception types, and we
/// may consider supporting other types (such as from key libraries utilized) at
/// a later point in time.
class Error {
public:
    Error(const std::string& message): message_(message) {}

    static Error From(std::exception_ptr eptr) {
        try {
            if (eptr) {
                std::rethrow_exception(eptr);
            }

            return Error("No Error");
        } catch(const std::exception& e) {
            return Error(e.what());
        }
    }

    static Error FromCurrentException() {
        return From(std::current_exception());
    }

    const std::string& message() const { return message_; }

private:
    std::string message_;
};

/// Common result type to communicate failures to caller. 
template <typename V, typename E = Error>
class Result {
public:
    using Value = V;
    using Error = E;

    //Result(): result_(Value {}) {}

    Result(const Result& other): result_(other.result_) { }
    Result(Result&& other) { std::swap(result_, other.result_); }

    Result(const Value& value): result_(value) {}
    Result(Value&& value): result_(value) {}
    Result(const Error& error): result_(error) {}
    Result(Error&& error): result_(error) {}

    Result& operator=(const Result& other) {
        result_ = other.result_;
        return *this;
    }

    Result& operator=(Result&& other) {
        result_ = std::forward(other.result_);
        return *this;
    }

    bool ok() const { return std::holds_alternative<Value>(result_); }
    bool failed() const { return std::holds_alternative<Error>(result_); }

    const Value& value() const { return std::get<Value>(result_); }
    const Error& error() const { return std::get<Error>(result_); }

private:
    std::variant<std::monostate, Value, Error> result_;
};

} // namespace slammer

#endif // ndef SLAMMER_SLAMMER_H