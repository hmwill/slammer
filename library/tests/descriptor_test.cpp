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
        typedef Descriptor Value;

        Descriptor operator() (const std::vector<const Descriptor*> descriptors) const {
            return Descriptor::ComputeCentroid(descriptors);
        }
    };

    DescriptorTree<CreateLeaf> tree;

    tree.ComputeTree(descriptors, 2);
    auto leaf = tree.FindNearest(descriptors[0]);
    EXPECT_TRUE(leaf.Get(0));
}

TEST(DescriptorTest, SearchTree) {
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
        typedef std::vector<Descriptor> Value;
        
        Value operator() (const std::vector<const Descriptor*> descriptors) const {
            Value result;
            std::transform(descriptors.begin(), descriptors.end(), std::back_inserter(result),
                           [](const auto& value) { return *value; });
            return result;
        }
    };

    DescriptorTree<CreateLeaf> tree;

    tree.ComputeTree(descriptors, 2);

    for (const auto& descriptor: descriptors) {
        auto leaf = tree.FindNearest(descriptor);

        // each leaf is a proper subset
        EXPECT_LT(leaf.size(), descriptors.size());
        
        // should have a direct match of the query descriptor 
        auto iter = std::find_if(leaf.begin(), leaf.end(), 
                                 [&](const auto &candidate) { 
                                    return Descriptor::ComputeDistance(candidate, descriptor) == 0;
                                 });

        EXPECT_NE(iter, leaf.end());
    }
}

namespace {

template <typename Int>
void SetBits(Descriptor& descriptor, Int bit) {
    descriptor.Set(bit);
}

template <typename Int, typename... Ints>
void SetBits(Descriptor& descriptor, Int bit, Ints... values) {
    descriptor.Set(bit);
    SetBits(descriptor, values...);
}

template <typename... Ints>
Descriptor MakeDescriptor(Ints... values) {
    Descriptor result;
    SetBits(result, values...);
    return result;
}

}

TEST(DescriptorTest, ComputeMatchesNoLimit) {
    Descriptors query, target;

    target.push_back(MakeDescriptor(1, 2, 3));
    target.push_back(MakeDescriptor(1, 2, 31, 32));
    target.push_back(MakeDescriptor(2, 31, 32, 40, 41));

    query.push_back(MakeDescriptor(1, 3, 31));
    query.push_back(MakeDescriptor(1, 3, 31, 32));
    query.push_back(MakeDescriptor(1, 3, 31, 32, 41));
    query.push_back(MakeDescriptor(2, 31, 32, 41));

    auto matches = ComputeMatches(target, query);
    EXPECT_EQ(matches.size(), 4);

    EXPECT_EQ(matches[0].query_index, 0);
    EXPECT_EQ(matches[0].target_index, 0);
    EXPECT_EQ(matches[0].distance, 2);

    EXPECT_EQ(matches[1].query_index, 1);
    EXPECT_EQ(matches[1].target_index, 1);
    EXPECT_EQ(matches[1].distance, 2);

    EXPECT_EQ(matches[2].query_index, 2);
    EXPECT_EQ(matches[2].target_index, 1);
    EXPECT_EQ(matches[2].distance, 3);

    EXPECT_EQ(matches[3].query_index, 3);
    EXPECT_EQ(matches[3].target_index, 2);
    EXPECT_EQ(matches[3].distance, 1);
}

TEST(DescriptorTest, ComputeMatchesWithLimit) {
    Descriptors query, target;

    target.push_back(MakeDescriptor(1, 2, 3));
    target.push_back(MakeDescriptor(1, 2, 31, 32));
    target.push_back(MakeDescriptor(2, 31, 32, 40, 41));

    query.push_back(MakeDescriptor(2, 31, 32, 41, 42));
    query.push_back(MakeDescriptor(1, 3, 31));
    query.push_back(MakeDescriptor(1, 3, 31, 32, 41));
    query.push_back(MakeDescriptor(1, 3, 31, 32, 41));

    auto matches = ComputeMatches(target, query, 2);
    EXPECT_EQ(matches.size(), 2);

    EXPECT_EQ(matches[0].query_index, 0);
    EXPECT_EQ(matches[0].target_index, 2);
    EXPECT_EQ(matches[0].distance, 2);

    EXPECT_EQ(matches[1].query_index, 1);
    EXPECT_EQ(matches[1].target_index, 0);
    EXPECT_EQ(matches[1].distance, 2);
}

TEST(DescriptorTest, ComputeKMatchesNoLimit) {
    Descriptors query, target;

    target.push_back(MakeDescriptor(1, 2, 3));
    target.push_back(MakeDescriptor(1, 2, 31, 32));
    target.push_back(MakeDescriptor(2, 31, 32, 40, 41));
    target.push_back(MakeDescriptor(1, 2, 5, 31, 32, 40, 41));

    query.push_back(MakeDescriptor(1, 3, 31));
    query.push_back(MakeDescriptor(1, 3, 31, 32));
    query.push_back(MakeDescriptor(1, 3, 31, 32, 41));
    query.push_back(MakeDescriptor(2, 31, 32, 41));

    auto match_list = ComputeKMatches(target, query, 3);
    EXPECT_EQ(match_list.size(), 4);

    for (const auto& matches: match_list) {
        EXPECT_EQ(matches.size(), 3);

        // distances in ascending order
        EXPECT_LE(matches[0].distance, matches[1].distance);
        EXPECT_LE(matches[1].distance, matches[2].distance);

        // query is the same
        EXPECT_EQ(matches[0].query_index, matches[1].query_index);
        EXPECT_EQ(matches[1].query_index, matches[2].query_index);

        // target is distinct
        EXPECT_NE(matches[0].target_index, matches[1].target_index);
        EXPECT_NE(matches[1].target_index, matches[2].target_index);
        EXPECT_NE(matches[0].target_index, matches[2].target_index);

        // target index is value
        EXPECT_LT(matches[0].target_index, target.size());
        EXPECT_LT(matches[1].target_index, target.size());
        EXPECT_LT(matches[2].target_index, target.size());
    }
}

TEST(DescriptorTest, ComputeKMatchesWithLimit) {
    Descriptors query, target;

    target.push_back(MakeDescriptor(1, 2, 3));
    target.push_back(MakeDescriptor(1, 2, 31, 32));
    target.push_back(MakeDescriptor(2, 31, 32, 40, 41));
    target.push_back(MakeDescriptor(1, 2, 5, 31, 32, 40, 41));

    query.push_back(MakeDescriptor(1, 3, 31));
    query.push_back(MakeDescriptor(1, 3, 31, 32));
    query.push_back(MakeDescriptor(1, 3, 31, 32, 41));
    query.push_back(MakeDescriptor(2, 31, 32, 41));

    auto match_list = ComputeKMatches(target, query, 3, false, 2);
    EXPECT_EQ(match_list.size(), 3);

    for (size_t match_index = 0; match_index < match_list.size(); ++match_index) {
        static const size_t match_index_size[] = { 1, 1, 2 };
        EXPECT_EQ(match_list[match_index].size(), match_index_size[match_index]);

        const auto& matches = match_list[match_index]; 

        for (size_t index = 0; index < matches.size(); ++index) {
            EXPECT_LE(matches[index].target_index, target.size());

            if (index > 0) {
                EXPECT_GE(matches[index].distance, matches[index - 1].distance);
                EXPECT_EQ(matches[index].query_index, matches[index - 1].query_index);
            } else {
                EXPECT_NE(matches[index].query_index, 2);
            }
        }
    }
}


TEST(DescriptorTest, ComputeKMatchesCrossNoLimit) {
    Descriptors query, target;

    target.push_back(MakeDescriptor(1, 3, 31));
    target.push_back(MakeDescriptor(1, 3, 31, 32));
    target.push_back(MakeDescriptor(1, 3, 31, 32, 41, 60, 100, 101, 102, 103, 104, 120));
    target.push_back(MakeDescriptor(2, 31, 32, 41, 60, 100, 101, 102, 103, 104, 120));

    query.push_back(MakeDescriptor(1, 2, 3, 60, 100, 101, 102, 103, 104, 120));
    query.push_back(MakeDescriptor(1, 2, 31, 32, 100, 101, 102, 103, 104, 120));
    query.push_back(MakeDescriptor(2, 31, 32, 40, 41));
    query.push_back(MakeDescriptor(1, 2, 5, 31, 32, 40, 41));
    query.push_back(MakeDescriptor(1, 2, 3));
    query.push_back(MakeDescriptor(1, 2, 31, 32));
    query.push_back(MakeDescriptor(2, 31, 32, 40, 41));
    query.push_back(MakeDescriptor(1, 2, 5, 31, 32, 40, 41));


    auto match_list = ComputeKMatches(target, query, 2, true);
    EXPECT_EQ(match_list.size(), 4);

    for (size_t match_index = 0; match_index < match_list.size(); ++match_index) {
        const auto& matches = match_list[match_index]; 
        static const size_t match_query_index[] = { 0, 1, 4, 5 };
        EXPECT_EQ(match_list[match_index][0].query_index, match_query_index[match_index]);
        EXPECT_EQ(matches.size(), 2);

        for (size_t index = 0; index < matches.size(); ++index) {
            EXPECT_LE(matches[index].target_index, target.size());

            if (index > 0) {
                EXPECT_GE(matches[index].distance, matches[index - 1].distance);
                EXPECT_EQ(matches[index].query_index, matches[index - 1].query_index);
            }
        }
    }
}
