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

#include <stdexcept>

#include <gtest/gtest.h>

#include "slammer/loris/driver.h"

using namespace slammer;
using namespace slammer::loris;

namespace {

template <typename Event>
class EventCapture {
public:
    EventCapture(size_t first, size_t max): first_(first), current_(0), max_(max) {}

    void Callback(const Event& event) {
        if (current_ >= first_ && events.size() < max_) {
            events.push_back(event);
        }

        ++current_;
    }

    std::vector<Event> events;

private:
    size_t first_, current_, max_;
};

class ImageChecker {
public:
    ImageChecker(size_t width, size_t height, size_t channels, int type):
        width_(width), height_(height), channels_(channels), type_(type) {}

    void Callback(const ImageEvent& event) {
        EXPECT_EQ(event.image.cols, width_);
        EXPECT_EQ(event.image.rows, height_);
        EXPECT_EQ(event.image.channels(), channels_);
        EXPECT_EQ(event.image.type(), type_);
    }

private:
    size_t width_, height_, channels_;
    int type_;
};

}

TEST(SlammerLorisTest, DriverTest) {
    using namespace std::placeholders;

    Driver driver("data/cafe1-1");

    EventCapture<AccelerometerEvent> d400_accelerometer(10, 20);
    driver.d400_accelerometer.AddHandler(std::bind(&EventCapture<AccelerometerEvent>::Callback, &d400_accelerometer, _1));

    EventCapture<GyroscopeEvent> d400_gyroscope(20, 10);
    driver.d400_gyroscope.AddHandler(std::bind(&EventCapture<GyroscopeEvent>::Callback, &d400_gyroscope, _1));

    ImageChecker color_checker(848, 480, 3, CV_8UC3);
    driver.color.AddHandler(std::bind(&ImageChecker::Callback, &color_checker, _1));

    ImageChecker depth_checker(848, 480, 1, CV_16UC1);
    driver.depth.AddHandler(std::bind(&ImageChecker::Callback, &depth_checker, _1));

    ImageChecker aligned_depth_checker(848, 480, 1, CV_16UC1);
    driver.aligned_depth.AddHandler(std::bind(&ImageChecker::Callback, &aligned_depth_checker, _1));

    // run for 2 secs of simulated events
    auto result = driver.Run(slammer::Timediff(2.0));
    EXPECT_TRUE(result.ok());

    EXPECT_EQ(d400_accelerometer.events.size(), 20);
    EXPECT_EQ(d400_gyroscope.events.size(), 10);
}