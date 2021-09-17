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

#include "slammer/loris/arrow_utils.h"
#include "slammer/loris/schema.h"

#include "arrow/io/api.h"

using namespace slammer;
using namespace slammer::loris;

TEST(SlammerLorisTests, ArrowUtils_ReadLorisTableMissingFile) {
    arrow::io::IOContext io_context = arrow::io::default_io_context();
    auto result = ReadLorisTable(std::string("Ivalid file name"), aligned_depth_schema,
                                 io_context);
    EXPECT_FALSE(result.ok());
    EXPECT_EQ(result.error().message(), std::string("Failed to open local file 'Ivalid file name'"));
}

TEST(SlammerLorisTests, ArrowUtils_ReadAlignedDepth) {
    arrow::io::IOContext io_context = arrow::io::default_io_context();
    auto result = ReadLorisTable(std::string("data/cafe1-1/aligned_depth.txt"), aligned_depth_schema,
                                 io_context);
    EXPECT_TRUE(result.ok());
    
    auto table = result.value();
    EXPECT_EQ(table->num_columns(), 2);
    EXPECT_EQ(table->num_rows(), 1708);

    auto columns = table->columns();
    EXPECT_EQ(columns.size(), 2);
    EXPECT_EQ(columns[0]->length(), 1708);
    EXPECT_EQ(columns[1]->length(), 1708);

    EXPECT_EQ(columns[0]->num_chunks(), 1);
    EXPECT_EQ(columns[1]->num_chunks(), 1);

    auto chunk0 = columns[0]->chunk(0);
    EXPECT_EQ(chunk0->length(), 1708);
    EXPECT_EQ(chunk0->type(), arrow::float64());

    auto element00 = chunk0->GetScalar(0);
    auto scalar00 = (*element00).get();
    EXPECT_EQ(dynamic_cast<arrow::DoubleScalar *>(scalar00)->value, 1560004885.446165);

    auto chunk1 = columns[1]->chunk(0);
    EXPECT_EQ(chunk1->length(), 1708);
    EXPECT_EQ(chunk1->type(), arrow::utf8());

    auto element10 = chunk1->GetScalar(0);
    auto scalar10 = (*element10).get();
    EXPECT_EQ(dynamic_cast<arrow::StringScalar *>(scalar10)->value->ToString(), 
              std::string("aligned_depth/1560004885.446165.png"));
}