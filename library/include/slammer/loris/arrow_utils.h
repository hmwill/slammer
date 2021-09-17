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

#ifndef SLAMMER_LORIS_ARROW_UTILS_H
#define SLAMMER_LORIS_ARROW_UTILS_H

#pragma once

#include "arrow/api.h"

#include "slammer/slammer.h"

namespace slammer {
namespace loris {

using DataTypePointer = std::shared_ptr<arrow::DataType>;
using TablePointer = std::shared_ptr<arrow::Table>;
using SchemaPointer = std::shared_ptr<arrow::Schema>;

/// Read a text data file from the LORIS data set into an Apache Arrow table.
///
/// \param path         The file location on disk
/// \param schema       The schema describing the columns of the data file
/// \param io_context   The arrow I/O context within which to perform this operation
///
/// \returns The resulting Table or and an error.
Result<TablePointer> ReadLorisTable(const std::string& path, const SchemaPointer& schema,
                                    const arrow::io::IOContext& io_context);

} // loris
} // namespace slammer

#endif //ndef SLAMMER_LORIS_ARROW_UTILS_H
