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

#include "slammer/image.h"

#include "boost/gil/io/write_view.hpp"
#include "boost/gil/extension/io/png.hpp"

using namespace slammer;

void 
FileImageLogger::LogImage(const boost::gil::gray8c_view_t image, 
                          const std::string& name) {
    auto path = GetPath(name, "png");
    boost::gil::write_view(path.string(), image, boost::gil::png_tag{});
}

void 
FileImageLogger::LogImage(const boost::gil::rgb8c_view_t image, 
                          const std::string& name) {
    auto path = GetPath(name, "png");
    boost::gil::write_view(path.string(), image, boost::gil::png_tag{});
}

std::filesystem::path 
FileImageLogger::GetPath(const std::string& name, const std::string& suffix) const {
    if (!std::filesystem::exists(prefix_)) {
        // TODO: Not sure yet about how to handle error situations here. For now, we just
        // eat the error silently, but will end up failing when the actual write operation is
        // called.
        std::error_code error;
        bool did_create = std::filesystem::create_directories(prefix_, error);
    }

    auto result = prefix_ / name;
    result.replace_extension(suffix);
    return result;
}