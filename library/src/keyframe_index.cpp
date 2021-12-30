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
        for (size_t index = 0; index < kNumBits; index += 8) {
            counters[index + 0] += descriptor->descriptor_[index + 0];
            counters[index + 1] += descriptor->descriptor_[index + 1];
            counters[index + 2] += descriptor->descriptor_[index + 2];
            counters[index + 3] += descriptor->descriptor_[index + 3];
            counters[index + 4] += descriptor->descriptor_[index + 4];
            counters[index + 5] += descriptor->descriptor_[index + 5];
            counters[index + 6] += descriptor->descriptor_[index + 6];
            counters[index + 7] += descriptor->descriptor_[index + 7];
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

Vocabulary::Vocabulary(): word_count_(0), random_engine_(kSeed) {
    FeatureDescriptors descriptors;
    descriptors.emplace_back();
    ComputeVocabulary(descriptors);
}

Vocabulary::Vocabulary(Vocabulary&& other) {
    std::swap(root_, other.root_);
    std::swap(word_count_, other.word_count_);
    std::swap(word_counts_, other.word_counts_);
    std::swap(word_weights_, other.word_weights_);
    std::swap(random_engine_, other.random_engine_);
}

Vocabulary::~Vocabulary() {}

void Vocabulary::ComputeVocabulary(const FeatureDescriptors& descriptors) {
    // Recursively partition the descriptors into sub-trees for each of the clusters
    root_ = ComputeSubtree(0, descriptors);

    // calculate scores
    for (auto count: word_counts_) {
        word_weights_.push_back(std::log(static_cast<double>(word_count_) / static_cast<double>(count)));
    }
}

Vocabulary::NodePointer Vocabulary::ComputeSubtree(size_t level, const FeatureDescriptors& descriptors) {
    if (level == kLevels || descriptors.size() < kArity) {
        word_counts_.push_back(descriptors.size());
        return std::make_unique<Node>(word_count_++);
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
            size_t min_distance_index = FindClosest(children, *descriptors[index]);

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

size_t Vocabulary::FindClosest(const Children& subtrees, const FeatureDescriptor& descriptor) {
    size_t min_distance_index = 0;
    FeatureDescriptor::Distance min_distance = 
        FeatureDescriptor::ComputeDistance(subtrees[0].centroid, descriptor);

    for (size_t center_index = 1; center_index < kArity; ++center_index) {
        FeatureDescriptor::Distance distance = 
            FeatureDescriptor::ComputeDistance(subtrees[center_index].centroid, descriptor);

        if (distance < min_distance) {
            min_distance = distance;
            min_distance_index = center_index;
        }
    }

    return min_distance_index;
}

std::unique_ptr<ImageDescriptor> Vocabulary::Encode(const FeatureDescriptor::Set& descriptors) const {
    auto result = std::make_unique<ImageDescriptor>();

    for (const auto& descriptor: descriptors) {
        auto word = FindWord(descriptor);
        result->descriptor_.try_emplace(word, 0);
        result->descriptor_[word] += word_weights_[word];
    }

    // normalize the scores
    double factor = 1.0 / std::accumulate(result->descriptor_.begin(), result->descriptor_.end(), 0.0, 
        [](double sum, const auto& item){return sum + item.second;});

    for (auto& item: result->descriptor_) {
        item.second *= factor;
    }

    return std::move(result);
}

const Vocabulary::Word Vocabulary::FindWord(const FeatureDescriptor& descriptor) const {
    const Node* node = root_.get();

    while (std::holds_alternative<Children>(node->node_type)) {
        const auto& children = std::get<Children>(node->node_type);
        auto child_index = FindClosest(children, descriptor);
        node = children[child_index].subtree.get();
    }

    return std::get<Word>(node->node_type);
}

//
// ImageDescriptor
//

ImageDescriptor::Score ImageDescriptor::Similarity(const ImageDescriptor& first, const ImageDescriptor& second) {
    auto iter_first = first.descriptor_.begin(), end_first = first.descriptor_.end();
    auto iter_second = second.descriptor_.begin(), end_second = second.descriptor_.end();

    Score score = 0.0;

    while (iter_first != end_first && iter_second != end_second) {
        if (iter_first->first == iter_second->first) {
            score += iter_first->second + iter_second->second - std::fabs(iter_first->second - iter_second->second);
            ++iter_first;
            ++iter_second;
        } else if (iter_first->first < iter_second->first) {
            iter_first = first.descriptor_.lower_bound(iter_second->first);
        } else {
            iter_second = second.descriptor_.lower_bound(iter_first->first);
        }
    }

    return score * 0.5;
}

//
// KeyframeIndex
//

KeyframeIndex::KeyframeIndex(Vocabulary&& vocabulary)
    : vocabulary_(std::move(vocabulary)), next_row_(0) {
    columns_.resize(vocabulary_.word_count());
}

KeyframeIndex::~KeyframeIndex() {}

void KeyframeIndex::Insert(const KeyframePointer& keyframe) {
    if (!keyframe->descriptor) {
        keyframe->descriptor = vocabulary_.Encode(FeatureDescriptor::From(keyframe->descriptions));
    }

    assert(reverse_index_.find(keyframe) == reverse_index_.end());

    RowIndex row_index;
    if (free_list_.empty()) {
        row_index = next_row_++;
    } else {
        row_index = free_list_.back();
        free_list_.pop_back();
    }

    reverse_index_[keyframe] = row_index;
    rows_[row_index] = Row { keyframe };

    for (const auto& item: keyframe->descriptor->descriptor_) {
        columns_[item.first].insert(row_index);
    }
}

void KeyframeIndex::Delete(const KeyframePointer& keyframe) {
    assert(keyframe->descriptor);
    
    auto reverse_row_iter = reverse_index_.find(keyframe);
    assert(reverse_row_iter != reverse_index_.end());
    auto row_index = reverse_row_iter->second;
    reverse_index_.erase(reverse_row_iter);

    rows_.erase(row_index);

    for (const auto& item: keyframe->descriptor->descriptor_) {
        columns_[item.first].erase(row_index);
    }

    free_list_.push_back(row_index);
}

void KeyframeIndex::Search(const KeyframePointer& query, std::vector<Result>& results,
                           size_t max_results) const {
    if (!query->descriptor) {
        query->descriptor = vocabulary_.Encode(FeatureDescriptor::From(query->descriptions));
    }

    absl::btree_set<RowIndex> excluded_rows;

    for (const auto& exclusion: query->covisible) {
        auto iter = reverse_index_.find(exclusion);

        if (iter != reverse_index_.end()) {
            excluded_rows.insert(iter->second);
        }
    }

    absl::btree_map<RowIndex, unsigned> word_counts;

    for (const auto& item: query->descriptor->descriptor_) {
        Vocabulary::Word word = item.first;
        const auto& column = columns_[word];

        for (auto row_index: column) {
            if (!excluded_rows.contains(row_index)) {
                word_counts[row_index] += word;
            }
        }
    }
    
    unsigned max_word_count = std::accumulate(word_counts.begin(), word_counts.end(), 0u,
                                              [](unsigned max, const auto& item) { return std::max(max, item.second); });
    unsigned min_word_count = static_cast<unsigned>(0.8 * max_word_count);
    std::vector<Result> result_candidates;

    for (const auto& item: word_counts) {
        if (item.second < min_word_count) {
            continue;
        }

        RowIndex row_index = item.first;
        const KeyframePointer& candidate = rows_.find(row_index)->second.keyframe;
        ImageDescriptor::Score score = ImageDescriptor::Similarity(*query->descriptor, *candidate->descriptor);

        result_candidates.emplace_back(Result(candidate, score));
        std::push_heap(result_candidates.begin(), result_candidates.end());
    }

    std::sort_heap(result_candidates.begin(), result_candidates.end());

    KeyframeSet ignore_set;

    for (auto iter = result_candidates.rbegin(); 
         iter != result_candidates.rend() && results.size() < max_results; ++iter) {

        if (ignore_set.find(iter->keyframe) != ignore_set.end()) {
            continue;
        }

        results.push_back(*iter);
        ignore_set.insert(iter->keyframe->covisible.begin(), iter->keyframe->covisible.end());
    }
}
