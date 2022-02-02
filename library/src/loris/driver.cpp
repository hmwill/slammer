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

#include "slammer/loris/driver.h"

#include <array>
#include <limits>

#include "boost/gil/extension/io/png.hpp"

using namespace slammer;
using namespace slammer::loris;

namespace {

inline bool IsLess(const HeapEntry& left, const HeapEntry& right) {
    return left.timestamp > right.timestamp;
}

} // namespace

namespace slammer {
namespace loris {

/// Common base class for event sources, which can be called by the driver.
/// Conceptually, an event source is a sequence of events along a common
/// time axis.
class AbstractEventSource {
public:
    AbstractEventSource(Driver& driver, const std::string& table_name, bool has_header):
        driver_(driver), table_name_(table_name), has_header_(has_header), row_index_(0) {}

    /// Initialize the event source
    virtual std::optional<Error> Initialize() = 0;

    /// Fire the next event and remove it from the event queue
    virtual std::optional<Error> FireEvent() = 0;

    std::optional<Timestamp> NextEventTime() const {
        if (row_index_ >= timestamp_column_.size()) {
            return std::optional<Timestamp>{};
        }

        return Timestamp { Timediff {  timestamp_column_[row_index_] } };
    }

protected:
    std::optional<Error> InitializeCommon() {
        std::string full_path = driver_.path() + "/" + table_name_ + ".txt";

        try {
            document_.Load(full_path, rapidcsv::LabelParams(-1, -1),
                           rapidcsv::SeparatorParams(' '), rapidcsv::ConverterParams(),
                           rapidcsv::LineReaderParams(true, '#'));
            timestamp_column_ = document_.GetColumn<double>(0);
        } catch (std::exception exc) {
            std::string message = "Error opening file '" + full_path + "'";
            return Error(message);
        }

        return std::optional<Error>{};
    }

    void NextRow() {
        ++row_index_;
    }

    Driver& driver_;
    std::string table_name_;
    bool has_header_;

    size_t row_index_;
    rapidcsv::Document document_;
    std::vector<double> timestamp_column_;
};

class AcceleratorEventSource: public AbstractEventSource {
public:
    AcceleratorEventSource(Driver& driver, const std::string& table_name, 
                           EventListenerList<AccelerometerEvent>& listeners):
        AbstractEventSource(driver, table_name, true), listeners_(listeners) {}

    virtual std::optional<Error> Initialize() override {
        auto result = InitializeCommon();

        if (!result.has_value()) {
            ax_ = document_.GetColumn<double>(1);
            ay_ = document_.GetColumn<double>(2);
            az_ = document_.GetColumn<double>(3);
        }

        return result;
    }

    virtual std::optional<Error> FireEvent() override {
        AccelerometerEvent event {
            Timestamp { Timediff {  timestamp_column_[row_index_] } },
            Vector3d {
                ax_[row_index_],
                ay_[row_index_],
                az_[row_index_]
            }
        };

        listeners_.HandleEvent(event);
        NextRow();

        return std::optional<Error>{};
    }

private:
    std::vector<double> ax_, ay_, az_;
    EventListenerList<AccelerometerEvent>& listeners_;
};

class GyroscopeEventSource: public AbstractEventSource {
public:
    GyroscopeEventSource(Driver& driver, const std::string& table_name, 
                         EventListenerList<GyroscopeEvent>& listeners):
        AbstractEventSource(driver, table_name, true), listeners_(listeners) {}

    virtual std::optional<Error> Initialize() override {
        auto result = InitializeCommon();

        if (!result.has_value()) {
            gx_ = document_.GetColumn<double>(1);
            gy_ = document_.GetColumn<double>(2);
            gz_ = document_.GetColumn<double>(3);
        }

        return result;
    }

    virtual std::optional<Error> FireEvent() override {
        GyroscopeEvent event {
            Timestamp { Timediff {  timestamp_column_[row_index_] } },
            Vector3d {
                gx_[row_index_],
                gy_[row_index_],
                gz_[row_index_]
            }
        };

        listeners_.HandleEvent(event);
        NextRow();

        return std::optional<Error>{};
    }

private:
    std::vector<double> gx_, gy_, gz_;
    EventListenerList<GyroscopeEvent>& listeners_;
};

class GroundtruthEventSource: public AbstractEventSource {
public:
    GroundtruthEventSource(Driver& driver, const std::string& table_name, 
                         EventListenerList<GroundtruthEvent>& listeners):
        AbstractEventSource(driver, table_name, true), listeners_(listeners) {}

    virtual std::optional<Error> Initialize() override {
        auto result = InitializeCommon();

        if (!result.has_value()) {
            px_ = document_.GetColumn<double>(1);
            py_ = document_.GetColumn<double>(2);
            pz_ = document_.GetColumn<double>(3);
            qx_ = document_.GetColumn<double>(4);
            qy_ = document_.GetColumn<double>(5);
            qz_ = document_.GetColumn<double>(6);
            qw_ = document_.GetColumn<double>(7);
        }

        return result;
    }

    virtual std::optional<Error> FireEvent() override {
        GroundtruthEvent event {
            Timestamp { Timediff {  timestamp_column_[row_index_] } },
            Vector3d {
                px_[row_index_],
                py_[row_index_],
                pz_[row_index_]
            },
            Quaterniond {
                qw_[row_index_],
                qx_[row_index_],
                qy_[row_index_],
                qz_[row_index_]
            }
        };

        listeners_.HandleEvent(event);
        NextRow();

        return std::optional<Error>{};
    }

private:
    std::vector<double> px_, py_, pz_;
    std::vector<double> qx_, qy_, qz_, qw_;
    EventListenerList<GroundtruthEvent>& listeners_;
};

template <typename EventType>
class ImageEventSource: public AbstractEventSource {
public:
    using Image = typename EventType::Image;
    using ImageInternal = typename Image::element_type;

    ImageEventSource(Driver& driver, const std::string& table_name, 
                     EventListenerList<EventType>& listeners):
        AbstractEventSource(driver, table_name, false), listeners_(listeners) {}

    virtual std::optional<Error> Initialize() override {
        auto result = InitializeCommon();

        if (!result.has_value()) {
            path_ = document_.GetColumn<std::string>(1);
        }

        return result;
    }

    virtual std::optional<Error> FireEvent() override {
        const auto& image_name = path_[row_index_];
        std::string full_path = driver_.path() + "/" + image_name;

        Image image(new ImageInternal());

        try {
            boost::gil::read_image(full_path, *image, boost::gil::png_tag{});
        } catch (...) {
            std::string error_message = "Could not read image file " + full_path;
            return Error(error_message);
        }

        EventType event {
            Timestamp { Timediff {  timestamp_column_[row_index_] } },
            std::move(image)
        };

        listeners_.HandleEvent(event);
        NextRow();

        return std::optional<Error>{};
    }

private:
    std::vector<std::string> path_;
    EventListenerList<EventType>& listeners_;
};

} // namespace loris
} // namespace slammer


Driver::Driver(const std::string& path)
    : path_(path) { }

Driver::~Driver() {

}

inline std::optional<Error> Driver::AddEventSource(std::unique_ptr<AbstractEventSource>&& event_source) {
    if (auto maybe_error = event_source->Initialize()) {
        return maybe_error;
    }

    if (auto optional_timestamp = event_source->NextEventTime()) {
        timestamp_heap_.emplace_back(HeapEntry { optional_timestamp.value(), std::move(event_source) });
    }

    return {};
}

Result<size_t> Driver::Run(const std::optional<Timediff> max_duration, 
                           const std::optional<size_t> max_num_events) {
    size_t max_processed_events = max_num_events.value_or(std::numeric_limits<size_t>::max());

    // Initialize all the event sources, including opening the underlying files

    // Add all the event sources to the heap
    AddEventSource(std::make_unique<AcceleratorEventSource>(*this, std::string("d400_accelerometer"), d400_accelerometer));
    AddEventSource(std::make_unique<GyroscopeEventSource>(*this, std::string("d400_gyroscope"), d400_gyroscope));
    AddEventSource(std::make_unique<ImageEventSource<DepthImageEvent>>(*this, std::string("aligned_depth"), aligned_depth));
    AddEventSource(std::make_unique<ImageEventSource<ColorImageEvent>>(*this, std::string("color"), color));
    AddEventSource(std::make_unique<ImageEventSource<DepthImageEvent>>(*this, std::string("depth"), depth));
    AddEventSource(std::make_unique<ImageEventSource<FisheyeImageEvent>>(*this, std::string("fisheye1"), fisheye1));
    AddEventSource(std::make_unique<ImageEventSource<FisheyeImageEvent>>(*this, std::string("fisheye2"), fisheye2));
    AddEventSource(std::make_unique<AcceleratorEventSource>(*this, std::string("t265_accelerometer"), t265_accelerometer));
    AddEventSource(std::make_unique<GyroscopeEventSource>(*this, std::string("t265_gyroscope"), t265_gyroscope));
    AddEventSource(std::make_unique<GroundtruthEventSource>(*this, std::string("groundtruth"), groundtruth));

    if (timestamp_heap_.empty()) {
        return 0;
    }

    std::make_heap(timestamp_heap_.begin(), timestamp_heap_.end(), IsLess);

    // Run the actual loop
    size_t processed_events = 0;
    Timestamp start_timestamp = timestamp_heap_[0].timestamp;

    while (!timestamp_heap_.empty() && 
           processed_events < max_processed_events) {
        std::pop_heap(timestamp_heap_.begin(), timestamp_heap_.end(), IsLess);

        if (timestamp_heap_.back().timestamp - start_timestamp >= max_duration) {
            break;
        }

        auto maybe_error = timestamp_heap_.back().event_source->FireEvent();
        if (maybe_error) {
            return maybe_error.value();
        }

        if (auto optional_timestamp = timestamp_heap_.back().event_source->NextEventTime()) {
            timestamp_heap_.back().timestamp = optional_timestamp.value();
            std::push_heap(timestamp_heap_.begin(), timestamp_heap_.end(), IsLess);
        } else {
            timestamp_heap_.pop_back();
        }
    }

    return processed_events;
}
