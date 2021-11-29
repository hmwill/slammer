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

#include "slammer/keyframe_index.h"

using namespace slammer;

//
// FeatureDescriptor
//

FeatureDescriptor FeatureDescriptor::ComputeCentroid(const std::vector<const FeatureDescriptor*> descriptors) {
    std::array<unsigned, kNumBits> counters;
    counters.fill(0);

    for (auto descriptor: descriptors) {
        for (size_t index = 0; index < kNumBits; ++index) {
            counters[index] += descriptor->descriptor_[index];
        }
    }

    unsigned threshold = descriptors.size() / 2 + descriptors.size() & 2;
    FeatureDescriptor result;

    for (size_t index = 0; index < kNumBits; ++index) {
        result.descriptor_[index] = counters[index] > threshold;
    }

    return result;
}

FeatureDescriptor::Distance FeatureDescriptor::ComputeDistance(const FeatureDescriptor& first, const FeatureDescriptor& second) {
    return (first.descriptor_ ^ second.descriptor_).count();
}

FeatureDescriptor::FeatureDescriptor(const uchar* bits) {
    for (auto block_count = kNumBits / sizeof(Bitset::block_type); block_count; --block_count) {
        Bitset::block_type block = 0;

        for (auto byte_count = sizeof(Bitset::block_type) / 8; byte_count; --byte_count) {
            block = (block << 8) | *bits++;
        }

        descriptor_.append(block);
    }
}

//
// Vocabulary
//

Vocabulary::Vocabulary(): word_count_(0), random_engine_(kSeed) {}
Vocabulary::~Vocabulary() {}

void Vocabulary::ComputeVocabulary(const FeatureDescriptors& descriptors) {
    // Recursively partition the descriptors into sub-trees for each of the clusters
    root_ = ComputeSubtree(0, descriptors);
}

Vocabulary::NodePointer Vocabulary::ComputeSubtree(size_t level, const FeatureDescriptors& descriptors) {
    if (level == kLevels || descriptors.size() < kArity) {
        Leaf leaf { 
            word_count_++, 
            static_cast<unsigned>(descriptors.size()) 
        };

        return std::make_unique<Node>(std::move(leaf));
    } 

    std::vector<FeatureDescriptor::Distance> min_distances;
    min_distances.reserve(descriptors.size());
    std::vector<size_t> assigned_cluster(descriptors.size(), 0);
    std::vector<size_t> prefix_sum_distance;
    std::array<size_t, kArity> cluster_center_indices;

    // pick the first cluster center
    std::uniform_int_distribution<size_t> distribution(0, descriptors.size() - 1);
    cluster_center_indices[0] = distribution(random_engine_);

    for (const auto& descriptor: descriptors) {
        min_distances.push_back(FeatureDescriptor::ComputeDistance(*descriptors[cluster_center_indices[0]],
                                                                   *descriptor));
    }

    // Iteratively determine the next cluster centers
    for (size_t index = 1; index < kArity; ++index) {
        prefix_sum_distance.clear();
        size_t total_weight = 0;

        for (auto distance: min_distances) {
            total_weight += distance * distance;
            prefix_sum_distance.push_back(total_weight);
        }

        std::uniform_int_distribution<size_t> distribution(0, total_weight - 1);

        auto split_point = std::max(static_cast<size_t>(1), distribution(random_engine_));
        auto split_iter = std::lower_bound(prefix_sum_distance.begin(), prefix_sum_distance.end(), split_point);
        assert(split_iter != prefix_sum_distance.end());

        size_t center_index = split_iter - prefix_sum_distance.begin();

        cluster_center_indices[index] = center_index;

        for (size_t descriptor_index = 0; descriptor_index != descriptors.size(); ++descriptor_index) {
            auto new_distance = FeatureDescriptor::ComputeDistance(*descriptors[center_index],
                                                                   *descriptors[descriptor_index]);

            if (new_distance < min_distances[descriptor_index]) {
                min_distances[descriptor_index] = new_distance;
                assigned_cluster[descriptor_index] = index;
            }
        }
    }

    Children children;
    std::array<FeatureDescriptors, kArity> partitions;

    // continue refining cluster assignments until we have convergence
    for (bool next_iteration = true; next_iteration;) {
        next_iteration = false;

        for (auto& partition: partitions) {
            partition.clear();
        }

        for (size_t index = 0; index < descriptors.size(); ++index) {
            partitions[assigned_cluster[index]].push_back(descriptors[index]);
        }

        for (size_t index = 0; index <kArity; ++index) {
            children[index].centroid = FeatureDescriptor::ComputeCentroid(partitions[index]);
        }

        for (size_t index = 0; index < descriptors.size(); ++index) {
            size_t min_distance_index = 0;
            FeatureDescriptor::Distance min_distance = 
                FeatureDescriptor::ComputeDistance(children[0].centroid, *descriptors[index]);

            for (size_t center_index = 1; center_index < kArity; ++center_index) {
                FeatureDescriptor::Distance distance = 
                    FeatureDescriptor::ComputeDistance(children[center_index].centroid, *descriptors[index]);

                if (distance < min_distance) {
                    min_distance = distance;
                    min_distance_index = center_index;
                }
            }

            if (assigned_cluster[index] != min_distance_index) {
                assigned_cluster[index] = min_distance_index;
                next_iteration = true;
            }
        }

    }

    for (size_t child_index = 0; child_index < kArity; ++child_index) {
        children[child_index].subtree = ComputeSubtree(level + 1, partitions[child_index]);
    }

    return std::make_unique<Node>(std::move(children));
}

//
// KeyframeIndex
//

KeyframeIndex::KeyframeIndex() {}
KeyframeIndex::~KeyframeIndex() {}

