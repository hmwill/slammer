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

#ifndef SLAMMER_DESCRIPTOR_H
#define SLAMMER_DESCRIPTOR_H

#pragma once

#include "slammer/slammer.h"

namespace slammer {

/// Representation of a binary feature descriptor; here, really, rotated BRIEF,
/// but the computation of the descriptor is hiddem within the feature detector.
///
/// Descriptors are fixed length bit strings and employ the Hamming distance as
/// natural pairwise distance function.
/// Description of a single image feature
///
/// For ORB features, which we are using, the feature descriptor is a 256-bit vector.
class Descriptor {
public:
    /// Number of bits per feature descriptor
    static const size_t kNumBits = 256;

    using Distance = size_t;
    using Collection = std::vector<Descriptor>;

    // We are using a dynamic bitset from the boost library until we have descriptor extraction
    // converted from not using OpenCV Mat types anymore.
    using Bitset = boost::dynamic_bitset<uint64_t>;

    Descriptor(): descriptor_(kNumBits) {}

    /// Set a specified descriptor bit
    ///
    /// \param index the index of the descriptor bit
    /// \param value the new bit value to set
    void Set(unsigned index, bool value = true) {
        descriptor_[index] = value;
    }

    /// Get a specified descriptor bit
    ///
    /// \param index the index of the descriptor bit
    bool Get(unsigned index) const {
        return descriptor_[index];
    }

    /// Ceate a feature descriptor based on the contents of a row in the given OpenCV matrix.
    /// The Matrix is expected to have elements of type CV_8UC1 and to have 32 columns.
    static Descriptor From(const cv::Mat& mat, int row)  {
        return Descriptor(mat.ptr(row));
    }   

    /// Ceate a feature descriptor based on the contents of all the rows in the given OpenCV matrix.
    /// The Matrix is expected to have elements of type CV_8UC1 and to have 32 columns.
    static Collection From(const cv::Mat& mat) {
        Collection result;

        for (int index = 0; index < mat.rows; ++index) {
            result.emplace_back(From(mat, index));
        }

        return result;
    }

    static void IntoPointerVector(const Collection& descriptors,
                                  std::vector<const Descriptor*>& result) {
        for (const auto& descriptor: descriptors) {
            result.push_back(&descriptor);
        }
    }

    /// Calculate the centroid of a collection of descriptors
    static Descriptor ComputeCentroid(const std::vector<const Descriptor*> descriptors);

    // Calculate the Hamming distance between two FeatureDescriptors
    static Distance ComputeDistance(const Descriptor& first, const Descriptor& second);

private:
    Descriptor(const uchar* bits);
    Descriptor(Bitset&& descriptor): descriptor_(descriptor) {}

    Bitset descriptor_;
};

struct Empty {};

template <typename Value>
struct Default {
    template <typename Arg>
    Value operator() (const Arg&) const { return Value{}; }
};

/// A tree data structure to support matching of descriptors. The structure of the tree
/// can be learned using a recursive k-means clustering method. The tree can also be
/// serialized and recreated to support peristing the tree structure across program
/// executions.
///
/// \tparam LeafData    Data associated with each leaf in the tree
template <typename LeafData = Empty, typename Func = Default<LeafData>, size_t kArity = 10>
class DescriptorTree {
public:
    typedef LeafData Value;
    typedef std::vector<Descriptor> Descriptors;

    /// Default constructor
    DescriptorTree()
        : random_engine_(kSeed) {};

    /// Calculate a the tree structure based on the distribution represented by the 
    /// provided collection of descriptors
    ///
    /// \param descriptors  a collection of descriptors from which the tree structure
    ///                     should be derived
    inline void ComputeTree(const Descriptors& descriptors, size_t max_level);

    /// Find the nearest leaf node in the tree data structure.
    ///
    /// \param descriptor the descriptor to locate in the tree
    ///
    /// \returns writable reference to the data stored within the leaf
    inline Value& FindNearest(const Descriptor& descriptor);

    /// Find the nearest leaf node in the tree data structure.
    ///
    /// \param descriptor the descriptor to locate in the tree
    ///
    /// \returns read-only reference to the data stored within the leaf
    inline const Value& FindNearest(const Descriptor& descriptor) const;

private:
    typedef std::vector<const Descriptor *> DescriptorPointers;

    struct Node;
    using NodePointer = std::unique_ptr<Node>;

    /// The number of tree levels
    static const size_t kLevels = 6;

    /// Random number generator seed
    static const int kSeed = 12345;

    // Information associated with a single child of a node
    struct Child {
        // the centroid defining the cluster node
        Descriptor centroid;

        // the associated sub-tree
        NodePointer subtree;
    };

    // The pointers of the root nodes of each sub-tree
    using Children = std::array<Child, kArity>;

    // Representation of a vocabulary tree node
    struct Node {
        Node(Value&& value): node_type(std::move(value)) {}
        Node(Children&& children): node_type(std::move(children)) {}

        std::variant<Value, Children> node_type;
    };

    // Recursive construction of the descriptor tree
    inline NodePointer ComputeSubtree(const DescriptorPointers& descriptors, size_t max_levels);

    // Create a leaf value, possibly using information about the descriptor set mapped to it
    inline Value CreateLeaf(const DescriptorPointers& Descriptors);

    // Determine the child tree whose median is closest to the query descriptor
    inline static size_t FindClosest(const Children& subtrees, const Descriptor& descriptor,
                                     size_t first_index = 0);

    // root of the descriptor tree
    NodePointer root_; 

    // Random number generator to use
    std::default_random_engine random_engine_;
};

template <typename LeafData, typename Func, size_t kArity>
void DescriptorTree<LeafData, Func, kArity>::ComputeTree(const Descriptors& descriptors, size_t max_levels) {
    DescriptorPointers pointers;
    std::transform(descriptors.begin(), descriptors.end(), 
                   std::back_inserter(pointers), [](const auto& descriptor) { return &descriptor; });
    root_ = ComputeSubtree(pointers, max_levels);
}

template <typename LeafData, typename Func, size_t kArity>
typename DescriptorTree<LeafData, Func, kArity>::Value& 
DescriptorTree<LeafData, Func, kArity>::FindNearest(const Descriptor& descriptor) {
    NodePointer * p_node = &root_;

    while (std::holds_alternative<Children>((*p_node)->node_type)) {
        auto& children = std::get<Children>((*p_node)->node_type);
        p_node = &children[FindClosest(children, descriptor)].subtree;
    }

    return std::get<Value>((*p_node)->node_type);
}

template <typename LeafData, typename Func, size_t kArity>
const typename DescriptorTree<LeafData, Func, kArity>::Value& 
DescriptorTree<LeafData, Func, kArity>::FindNearest(const Descriptor& descriptor) const {
    const NodePointer * p_node = &root_;

    while (std::holds_alternative<Children>((*p_node)->node_type)) {
        const auto& children = std::get<Children>((*p_node)->node_type);
        p_node = &children[FindClosest(children, descriptor)].subtree;
    }

    return std::get<Value>((*p_node)->node_type);
}

template <typename LeafData, typename Func, size_t kArity>
typename DescriptorTree<LeafData, Func, kArity>::NodePointer 
DescriptorTree<LeafData, Func, kArity>::ComputeSubtree(const DescriptorPointers& descriptors, size_t max_levels) {
    if (!max_levels || descriptors.size() < kArity) {
        return std::make_unique<Node>(CreateLeaf(descriptors));
    } 

    std::vector<Descriptor::Distance> min_distances;
    min_distances.reserve(descriptors.size());
    std::vector<size_t> assigned_cluster(descriptors.size(), 0);
    std::vector<size_t> prefix_sum_distance;
    std::array<size_t, kArity> cluster_center_indices;

    // pick the first cluster center
    std::uniform_int_distribution<size_t> distribution(0, descriptors.size() - 1);
    cluster_center_indices[0] = distribution(random_engine_);

    for (const auto& descriptor: descriptors) {
        min_distances.push_back(Descriptor::ComputeDistance(*descriptors[cluster_center_indices[0]],
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
            auto new_distance = Descriptor::ComputeDistance(*descriptors[center_index],
                                                                   *descriptors[descriptor_index]);

            if (new_distance < min_distances[descriptor_index]) {
                min_distances[descriptor_index] = new_distance;
                assigned_cluster[descriptor_index] = index;
            }
        }
    }

    Children children;
    std::array<DescriptorPointers, kArity> partitions;

    // continue refining cluster assignments until we have convergence
    for (bool next_iteration = true; next_iteration;) {
        next_iteration = false;

        for (auto& partition: partitions) {
            partition.clear();
        }

        for (size_t index = 0; index < descriptors.size(); ++index) {
            partitions[assigned_cluster[index]].push_back(descriptors[index]);
        }

        for (size_t index = 0; index < partitions.size(); ++index) {
            children[index].centroid = Descriptor::ComputeCentroid(partitions[index]);
        }

        for (size_t index = 0; index < descriptors.size(); ++index) {
            size_t min_distance_index = FindClosest(children, *descriptors[index], assigned_cluster[index]);

            if (assigned_cluster[index] != min_distance_index) {
                assigned_cluster[index] = min_distance_index;
                next_iteration = true;
            }
        }

    }

    for (size_t child_index = 0; child_index < kArity; ++child_index) {
        children[child_index].subtree = ComputeSubtree(partitions[child_index], max_levels - 1);
    }

    return std::make_unique<Node>(std::move(children));
}

template <typename LeafData, typename Func, size_t kArity>
typename DescriptorTree<LeafData, Func, kArity>::Value 
DescriptorTree<LeafData, Func, kArity>::CreateLeaf(const DescriptorPointers& descriptors) {
    Func func;
    return func(descriptors);
}

template <typename LeafData, typename Func, size_t kArity>
size_t DescriptorTree<LeafData, Func, kArity>::FindClosest(const Children& subtrees, const Descriptor& descriptor, size_t first_index) {
    size_t min_distance_index = first_index;
    Descriptor::Distance min_distance = 
        Descriptor::ComputeDistance(subtrees[first_index].centroid, descriptor);

    for (size_t center_index = (first_index + 1) % subtrees.size(); center_index != first_index; center_index = (center_index + 1) % subtrees.size()) {
        Descriptor::Distance distance = 
            Descriptor::ComputeDistance(subtrees[center_index].centroid, descriptor);

        if (distance < min_distance) {
            min_distance = distance;
            min_distance_index = center_index;
        }
    }

    return min_distance_index;
}

} // namespace slammer

#endif //ndef SLAMMER_DESCRIPTOR_H
