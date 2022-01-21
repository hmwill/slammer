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

#include "slammer/descriptor.h"

using namespace slammer;

TEST(DescriptorTest, Distance) {
    Descriptor descriptor1, descriptor2;

    EXPECT_EQ(Descriptor::ComputeDistance(descriptor1, descriptor2), 0);

    descriptor1.Set(3);
    descriptor1.Set(73);
    EXPECT_EQ(Descriptor::ComputeDistance(descriptor1, descriptor2), 2);

    descriptor2.Set(125);
    EXPECT_EQ(Descriptor::ComputeDistance(descriptor1, descriptor2), 3);

    descriptor1.Set(73, false);
    descriptor1.Set(125);
    EXPECT_EQ(Descriptor::ComputeDistance(descriptor1, descriptor2), 1);
}

TEST(DescriptorTest, BuildTree) {
    std::vector<Descriptor> descriptors;

    for (size_t cluster = 0; cluster < 10; ++cluster) {
        size_t cluster_offset = cluster * 25;
        size_t group_offset = cluster_offset + 10;

        for (size_t member = 0; member < 8; ++member) {
            Descriptor descriptor;

            descriptor.Set(cluster_offset);
            descriptor.Set(cluster_offset + 1);
            descriptor.Set(cluster_offset + 2);
            descriptor.Set(cluster_offset + 3);
            descriptor.Set(cluster_offset + 4);
            descriptor.Set(cluster_offset + 5);
            descriptor.Set(cluster_offset + 6);


            descriptor.Set(group_offset + (member) % 15);
            descriptor.Set(group_offset + (member + 2) % 15);
            descriptor.Set(group_offset + (member + 4) % 15);

            descriptors.emplace_back(descriptor);
        }
    }

    struct CreateLeaf {
        Descriptor operator() (const std::vector<const Descriptor*> descriptors) const {
            return Descriptor::ComputeCentroid(descriptors);
        }
    };

    DescriptorTree<Descriptor, CreateLeaf> tree;

    tree.ComputeTree(descriptors, 2);
    auto leaf = tree.FindNearest(descriptors[0]);
    EXPECT_TRUE(leaf.Get(0));
}