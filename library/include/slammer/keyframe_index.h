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

#ifndef SLAMMER_KEYFRAME_INDEX_H
#define SLAMMER_KEYFRAME_INDEX_H

#pragma once

#include "slammer/slammer.h"

#include "absl/container/btree_set.h"
#include "absl/container/btree_map.h"

#include "slammer/map.h"

namespace slammer {

/// Description of a single image feature
///
/// For ORB features, which we are using, the feature descriptor is a 256-bit vector.
class FeatureDescriptor {
public:
    /// Number of bits per feature descriptor
    static const size_t kNumBits = 256;

    using Distance = size_t;
    using Set = std::vector<FeatureDescriptor>;

    // We are using a dynamic bitset from the boost library until we have descriptor extraction
    // converted from not using OpenCV Mat types anymore.
    using Bitset = boost::dynamic_bitset<uint64_t>;

    FeatureDescriptor(): descriptor_(kNumBits) {}

    /// Ceate a feature descriptor based on the contents of a row in the given OpenCV matrix.
    /// The Matrix is expected to have elements of type CV_8UC1 and to have 32 columns.
    static FeatureDescriptor From(const cv::Mat& mat, int row)  {
        return FeatureDescriptor(mat.ptr(row));
    }   

    /// Ceate a feature descriptor based on the contents of all the rows in the given OpenCV matrix.
    /// The Matrix is expected to have elements of type CV_8UC1 and to have 32 columns.
    static Set From(const cv::Mat& mat) {
        Set result;

        for (int index = 0; index < mat.rows; ++index) {
            result.emplace_back(From(mat, index));
        }

        return result;
    }

    static void IntoPointerVector(const Set& descriptors,
                                  std::vector<const FeatureDescriptor*>& result) {
        for (const auto& descriptor: descriptors) {
            result.push_back(&descriptor);
        }
    }

    /// Calculate the centroid of a collection of descriptors
    static FeatureDescriptor ComputeCentroid(const std::vector<const FeatureDescriptor*> descriptors);

    // Calculate the Hamming distance between two FeatureDescriptors
    static Distance ComputeDistance(const FeatureDescriptor& first, const FeatureDescriptor& second);

private:
    FeatureDescriptor(const uchar* bits);
    FeatureDescriptor(Bitset&& descriptor): descriptor_(descriptor) {}

    Bitset descriptor_;
};

class ImageDescriptor;

/// Representation of the abstract "words" used to describe images.
///
/// The vocabulary maps the overall set of possible features to a predetermined
/// collection of words, which are represented each by an integer value.
/// Internally, the vocabulary is represented as n-ary tree structure, where each
/// node is a cluster center and root of a sub-tree. A given feature descriptor is
/// mapped to the node and its sub-tree if it is the nearest neighbor among all
/// nodes at the level.
class Vocabulary {
public:
    /// Representation of a single word in the vocabulary
    using Word = uint32_t;
    using FeatureDescriptors = std::vector<const FeatureDescriptor*>;

    Vocabulary();
    Vocabulary(Vocabulary&& other);

    ~Vocabulary();

    /// Create a vocabulary based on a collection of feature descriptors
    ///
    /// Should this be feature descriptors, or a collection of collections of feature descriptors?
    /// This mostly affects the word frequency calculation (relative to feature count vs. relative to frame count)
    void ComputeVocabulary(const FeatureDescriptors& descriptors);

    /// Encode a set of feature descriptors using the vocabulary as imaeg descriptor
    ///
    /// \param descriptors the feature descriptors to encode via the vocabulary
    std::unique_ptr<ImageDescriptor> Encode(const FeatureDescriptor::Set& descriptors) const;

private:
    struct Node;
    using NodePointer = std::unique_ptr<Node>;

    /// Each node has this number of children
    static const size_t kArity = 10;

    /// The number of tree levels
    static const size_t kLevels = 6;

    /// Random number generator seed
    static const int kSeed = 12345;

    // Information associated with a single child of a node
    struct Child {
        // the centroid defining the cluster node
        FeatureDescriptor centroid;

        // the associated sub-tree
        NodePointer subtree;
    };

    // The pointers of the root nodes of each sub-tree
    using Children = std::array<Child, kArity>;

    // Representation of a vocabulary tree node
    struct Node {
        Node(Word word): node_type(word) {}
        Node(Children&& children): node_type(std::move(children)) {}

        std::variant<Word, Children> node_type;
    };

    NodePointer ComputeSubtree(size_t level, const FeatureDescriptors& descriptors);

    static size_t FindClosest(const Children& subtrees, const FeatureDescriptor& descriptor);

    const Word FindWord(const FeatureDescriptor& descriptor) const;

    // root of the vocabulary tree
    NodePointer root_; 

    // total number of words in the vocabulary
    Word word_count_;

    // number of occurrences of each word in training corpus
    std::vector<unsigned> word_counts_;

    // word weight based on inverse document frequency
    std::vector<double> word_weights_;

    /// Random number generator to use
    std::default_random_engine random_engine_;
};

/// Description of a single image as a bag of words, that is, for all features that are
/// present in the image, the occurrence counts of the corresponding words.
class ImageDescriptor {
public:
    using WordWeight = double;
    using Score = double;

    friend class Vocabulary;
    friend class KeyframeIndex;

    /// Calculate a similarity score for two image decriptors
    static Score Similarity(const ImageDescriptor& first, const ImageDescriptor& second);

private:
    absl::btree_map<Vocabulary::Word, WordWeight> descriptor_;
};


/// Feature-based index of keyframes which can retrieve keyfranes based on image 
/// similarity.
class KeyframeIndex {
public:
    KeyframeIndex(Vocabulary&& vocabulary);
    ~KeyframeIndex();

    // no copy construction, no assignment
    KeyframeIndex(const KeyframeIndex&) = delete;
    KeyframeIndex(KeyframeIndex&&) = delete;

    KeyframeIndex& operator=(const KeyframeIndex&) = delete;
    KeyframeIndex& operator=(KeyframeIndex&&) = delete;

    /// Representation of a search result
    struct Result {
        /// pointer to the keyframe
        KeyframePointer keyframe;

        /// the associated match score
        ImageDescriptor::Score score;

        Result(const KeyframePointer& pointer, ImageDescriptor::Score value): keyframe(pointer), score(value) {}
    };

    /// Insert a keyframe into the search index.
    //
    // Should the descriptor become a member variable of the keyframe???
    void Insert(const KeyframePointer& keyframe);
    
    /// Remove a keyframe from the search index.
    void Delete(const KeyframePointer& keyframe);

    /// Retrieve the keyframes that best match the given query descriptor
    ///
    /// \param query the keyframe to be matched by the search results
    /// \param results a container that will receive the search results
    /// \param max_results the maximum number of search results to return
    void Search(const KeyframePointer& query, std::vector<Result>& results,
                size_t max_results = 10) const;

    Vocabulary& vocabulary() { return vocabulary_; }
    const Vocabulary& vocabulary() const { return vocabulary_; }

private:
    using RowIndex = size_t;
    using Column = absl::btree_set<RowIndex>;
    using Columns = std::vector<Column>;

    // Inverted index; for each Word in the vocabulary we have a column vector for row indices
    Columns columns_;

    struct Row {
        KeyframePointer keyframe;
    };

    // Collection of keyframes
    absl::btree_map<RowIndex, Row> rows_;

    // Free list of empty slots
    std::vector<RowIndex> free_list_;

    // Reverse index
    std::unordered_map<KeyframePointer, RowIndex> reverse_index_;

    // Next insert location
    RowIndex next_row_;

    // the vocabulary used by the index
    Vocabulary vocabulary_;
};

} // namespace slammer

#endif //ndef SLAMMER_KEYFRAME_INDEX_H
