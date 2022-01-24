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
// Vocabulary
//

Vocabulary::Vocabulary(): tree_(State()) {
    Descriptors descriptors;
    descriptors.emplace_back(Descriptor());
    tree_.ComputeTree(descriptors, kMaxDepth);
}

// Vocabulary::Vocabulary(Vocabulary&& other) {
//     std::swap(root_, other.root_);
//     std::swap(word_count_, other.word_count_);
//     std::swap(word_counts_, other.word_counts_);
//     std::swap(word_weights_, other.word_weights_);
//     std::swap(random_engine_, other.random_engine_);
// }

Vocabulary::~Vocabulary() {}

void Vocabulary::ComputeVocabulary(const Descriptor::Collection& descriptors) {
    // Recursively partition the descriptors into sub-trees for each of the clusters
    tree_.ComputeTree(descriptors, kMaxDepth);

    auto& data = state();

    // calculate scores
    for (auto count: data.word_counts) {
        data.word_weights.push_back(std::log(static_cast<double>(data.word_count) / static_cast<double>(count)));
    }
}

std::unique_ptr<ImageDescriptor> Vocabulary::Encode(const Descriptor::Collection& descriptors) const {
    auto result = std::make_unique<ImageDescriptor>();
    const auto& data = state();

    for (const auto& descriptor: descriptors) {
        auto word = FindWord(descriptor);
        result->descriptor_.try_emplace(word, 0);
        result->descriptor_[word] += data.word_weights[word];
    }

    // normalize the scores
    double factor = 1.0 / std::accumulate(result->descriptor_.begin(), result->descriptor_.end(), 0.0, 
        [](double sum, const auto& item){return sum + item.second;});

    for (auto& item: result->descriptor_) {
        item.second *= factor;
    }

    return std::move(result);
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
        keyframe->descriptor = vocabulary_.Encode(Descriptor::From(keyframe->descriptions));
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
        query->descriptor = vocabulary_.Encode(Descriptor::From(query->descriptions));
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
