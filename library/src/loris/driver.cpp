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
#include "slammer/loris/arrow_utils.h"
#include "slammer/loris/schema.h"

#include <array>
#include <limits>

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
    AbstractEventSource(Driver& driver, const std::string& table_name):
        driver_(driver), table_name_(table_name) {}

    /// Initialize the event source
    virtual std::optional<Error> Initialize() = 0;

    /// Fire the next event and remove it from the event queue
    virtual std::optional<Error> FireEvent() = 0;

    std::optional<Timestamp> NextEventTime() const {
        if (!remaining_) {
            return std::optional<Timestamp>{};
        }

        auto chunk = columns_[0]->chunk(chunk_indices[0]);
        auto double_array = dynamic_cast<arrow::NumericArray<arrow::DoubleType> *>(chunk.get());

        return Timestamp { Timediff {  double_array->Value(array_indices[0]) } };
    }

    double GetDoubleArrayValue(int column_index, int& chunk_index, int& array_index) {
        auto array = 
            dynamic_cast<arrow::NumericArray<arrow::DoubleType> *>(columns_[column_index]->chunk(chunk_index).get());
        auto result = array->Value(array_indices[0]);

        if (++array_index >= array->length()) {
            array_index = 0;
            ++chunk_index;
        }

        return result;
    }

    Timestamp GetTimestampArrayValue(int column_index, int& chunk_index, int& array_index) {
        return Timestamp { Timediff { GetDoubleArrayValue(column_index, chunk_index, array_index) } };
    }

    std::string GetStringArrayValue(int column_index, int& chunk_index, int& array_index) {
        auto array = 
            dynamic_cast<arrow::BinaryArray *>(columns_[column_index]->chunk(chunk_index).get());
        auto result = array->GetString(array_indices[0]);

        if (++array_index >= array->length()) {
            array_index = 0;
            ++chunk_index;
        }

        return result;
    }

protected:
    std::optional<Error> InitializeCommon(const SchemaPointer& schema) {
        std::string full_path = driver_.path() + "/" + table_name_ + ".txt";
        auto maybe_table = ReadLorisTable(full_path, schema, driver_.io_context());

        if (maybe_table.ok()) {
            auto table = maybe_table.value();
            remaining_ = table->num_rows();
            columns_ = table->columns();
            chunk_indices.resize(columns_.size());
            array_indices.resize(columns_.size());
        } else {
            return maybe_table.error();
        }

        return std::optional<Error>{};
    }

    Driver& driver_;
    std::string table_name_;
    size_t remaining_;
    std::vector<std::shared_ptr<arrow::ChunkedArray>> columns_;
    std::vector<int> chunk_indices;
    std::vector<int> array_indices;
};

class AcceleratorEventSource: public AbstractEventSource {
public:
    AcceleratorEventSource(Driver& driver, const std::string& table_name, 
                           EventListenerList<AccelerometerEvent>& listeners):
        AbstractEventSource(driver, table_name), listeners_(listeners) {}

    virtual std::optional<Error> Initialize() override {
        return InitializeCommon(accelerometer_schema);
    }

    virtual std::optional<Error> FireEvent() override {
        AccelerometerEvent event {
            GetTimestampArrayValue(0, chunk_indices[0], array_indices[0]),
            Vector3d {
                GetDoubleArrayValue(1, chunk_indices[1], array_indices[1]),
                GetDoubleArrayValue(2, chunk_indices[2], array_indices[2]),
                GetDoubleArrayValue(3, chunk_indices[3], array_indices[3])
            }
        };

        listeners_.HandleEvent(event);
        --remaining_;

        return std::optional<Error>{};
    }

private:

    EventListenerList<AccelerometerEvent>& listeners_;
};

class GyroscopeEventSource: public AbstractEventSource {
public:
    GyroscopeEventSource(Driver& driver, const std::string& table_name, 
                         EventListenerList<GyroscopeEvent>& listeners):
        AbstractEventSource(driver, table_name), listeners_(listeners) {}

    virtual std::optional<Error> Initialize() override {
        return InitializeCommon(gyroscope_schema);
    }

    virtual std::optional<Error> FireEvent() override {
        GyroscopeEvent event {
            GetTimestampArrayValue(0, chunk_indices[0], array_indices[0]),
            Vector3d {
                GetDoubleArrayValue(1, chunk_indices[1], array_indices[1]),
                GetDoubleArrayValue(2, chunk_indices[2], array_indices[2]),
                GetDoubleArrayValue(3, chunk_indices[3], array_indices[3])
            }
        };

        listeners_.HandleEvent(event);
        --remaining_;

        return std::optional<Error>{};
    }

private:
    EventListenerList<GyroscopeEvent>& listeners_;
};

class ImageEventSource: public AbstractEventSource {
public:
    ImageEventSource(Driver& driver, const std::string& table_name, 
                     EventListenerList<ImageEvent>& listeners):
        AbstractEventSource(driver, table_name), listeners_(listeners) {}

    virtual std::optional<Error> Initialize() override {
        return InitializeCommon(image_schema);
    }

    virtual std::optional<Error> FireEvent() override {
        auto image_name = GetStringArrayValue(1, chunk_indices[1], array_indices[1]);
        std::string full_path = driver_.path() + "/" + image_name;
        auto image = cv::imread(full_path, cv::IMREAD_UNCHANGED);

        if (image.empty()) {
            std::string error_message = "Could not read image file " + full_path;
            return Error(error_message);
        }

        ImageEvent event {
            GetTimestampArrayValue(0, chunk_indices[0], array_indices[0]),
            image
        };

        listeners_.HandleEvent(event);
        --remaining_;

        return std::optional<Error>{};
    }

private:
    EventListenerList<ImageEvent>& listeners_;
};

} // namespace loris
} // namespace slammer


Driver::Driver(const std::string& path, arrow::io::IOContext io_context):
    path_(path), io_context_(io_context)
{ }

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
    AddEventSource(std::make_unique<ImageEventSource>(*this, std::string("aligned_depth"), aligned_depth));
    AddEventSource(std::make_unique<ImageEventSource>(*this, std::string("color"), color));
    AddEventSource(std::make_unique<ImageEventSource>(*this, std::string("depth"), depth));
    AddEventSource(std::make_unique<ImageEventSource>(*this, std::string("fisheye1"), fisheye1));
    AddEventSource(std::make_unique<ImageEventSource>(*this, std::string("fisheye2"), fisheye2));
    AddEventSource(std::make_unique<AcceleratorEventSource>(*this, std::string("t265_accelerometer"), t265_accelerometer));
    AddEventSource(std::make_unique<GyroscopeEventSource>(*this, std::string("t265_gyroscope"), t265_gyroscope));

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
