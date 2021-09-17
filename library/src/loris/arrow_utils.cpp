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

#include "slammer/loris/arrow_utils.h"

#include "arrow/csv/api.h"
#include "arrow/filesystem/api.h"


using namespace slammer;
using namespace slammer::loris;

namespace {

std::vector<std::string> GetColumnNames(const SchemaPointer &schema) {
    std::vector<std::string> result;
    auto num_fields = schema->num_fields();

    for (int index = 0; index < num_fields; ++index) {
        result.push_back(schema->field(index)->name());
    }

    return result;
}

std::unordered_map<std::string, DataTypePointer> GetColumnTypes(const SchemaPointer& schema) {
    std::unordered_map<std::string, DataTypePointer> result;
    auto num_fields = schema->num_fields();

    for (int index = 0; index < num_fields; ++index) {
        result[schema->field(index)->name()] = schema->field(index)->type();
    }

    return result;
}

} // namespace

Result<TablePointer> 
slammer::loris::ReadLorisTable(const std::string &path, const SchemaPointer &schema,
                               const arrow::io::IOContext &io_context) {
    auto localfs = std::make_shared<arrow::fs::LocalFileSystem>();
    auto maybe_input = localfs->OpenInputStream(path);

    if (!maybe_input.ok()) {
        return maybe_input.status().message();
    }

    std::shared_ptr<arrow::io::InputStream> input = *maybe_input;

    auto read_options = arrow::csv::ReadOptions::Defaults();
    read_options.column_names = GetColumnNames(schema);
    read_options.autogenerate_column_names = false;
    
    auto parse_options = arrow::csv::ParseOptions::Defaults();
    parse_options.delimiter = ' ';

    auto convert_options = arrow::csv::ConvertOptions::Defaults();
    convert_options.column_types = GetColumnTypes(schema);

    // Instantiate TableReader from input stream and options
    auto maybe_reader =
        arrow::csv::TableReader::Make(io_context,
                                      input,
                                      read_options,
                                      parse_options,
                                      convert_options);
    if (!maybe_reader.ok()) {
        return maybe_reader.status().message();
    }
    std::shared_ptr<arrow::csv::TableReader> reader = *maybe_reader;

    // Read table from CSV file
    auto maybe_table = reader->Read();
    if (!maybe_table.ok()) {
        return maybe_input.status().message();
    }

    return *maybe_table;
}