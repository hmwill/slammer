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

    // We are using a dynamic bitset from the boost library until we have descriptor extraction
    // converted from not using OpenCV Mat types anymore.
    using Bitset = boost::dynamic_bitset<uint64_t>;

    FeatureDescriptor(): descriptor_(kNumBits) {}

    /// Ceate a feature descriptor based on the contents of a row in the given OpenCV matrix.
    /// The Matrix is expected to have elements of type CV_8UC1 and to have 32 columns.
    static FeatureDescriptor From(const cv::Mat& mat, int row)  {
        return FeatureDescriptor(mat.ptr(row));
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
    ~Vocabulary();

    /// Create a vocabulary based on a collection of feature descriptors
    ///
    /// Should this be feature descriptors, or a collection of collections of feature descriptors?
    /// This mostly affects the word frequency calculation (relative to feature count vs. relative to frame count)
    void ComputeVocabulary(const FeatureDescriptors& descriptors);

    /// Boost serialization support
    template<class Archive>
    void serialize(Archive & ar, const unsigned int /* file_version */);

private:
    struct Node;
    using NodePointer = std::unique_ptr<Node>;

    NodePointer ComputeSubtree(size_t level, const FeatureDescriptors& descriptors);

    /// Each node has this number of children
    static const size_t kArity = 10;

    /// The number of tree levels
    static const size_t kLevels = 6;

    /// Random number generator seed
    static const int kSeed = 12345;

    // Fields that are specific to a leaf node
    struct Leaf {
        // the identifier that will be used to represent the word associated with this leaf
        Word word;

        // the observed frequency of the word in the training corpus (number of images having an occurence of the word)
        unsigned frequency;
    };

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
        Node(Leaf&& leaf): node_type(std::move(leaf)) {}
        Node(Children&& children): node_type(std::move(children)) {}

        std::variant<Leaf, Children> node_type;
    };

    // root of the vocabulary tree
    NodePointer root_; 

    // total number of words in the vocabulary
    Word word_count_;

    /// Random number generator to use
    std::default_random_engine random_engine_;
};

/// Description of a single image as a bag of words, that is, for all features that are
/// present in the image, the occurrence counts of the corresponding words.
class ImageDescriptor {
public:
    using WordCount = uint32_t;

private:
    std::map<Vocabulary::Word, WordCount> descriptor_;
};

/// Feature-based index of keyframes which can retrieve keyfranes based on image 
/// similarity.
class KeyframeIndex {
public:
    KeyframeIndex();
    ~KeyframeIndex();

    // no copy construction, no assignment
    KeyframeIndex(const KeyframeIndex&) = delete;
    KeyframeIndex(KeyframeIndex&&) = delete;

    KeyframeIndex& operator=(const KeyframeIndex&) = delete;
    KeyframeIndex& operator=(KeyframeIndex&&) = delete;

    /// Representation of a search result
    class Result {
        /// pointer to the keyframe
        KeyframePointer keyframe;

        /// the associated match score
        double score;
    };

    /// Insert a keyframe into the search index.
    //
    // Should the descriptor become a member variable of the keyframe???
    void Insert(const ImageDescriptor& descriptor, const KeyframePointer& keyframe);
    
    /// Remove a keyframe from the search index.
    void Delete(const KeyframePointer& keyframe);

    /// Retrieve the keyframes that best match the given query descriptor
    ///
    /// \param query the image descriptor to be matched by the search results
    /// \param results a container that will receive the search results
    /// \param max_results the maximum number of search results to return
    void Search(const ImageDescriptor& query, std::vector<Result>& results,
                size_t max_results = 10) const;

private:
};

} // namespace slammer

#endif //ndef SLAMMER_KEYFRAME_INDEX_H
