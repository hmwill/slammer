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

#include "slammer/descriptor.h"

using namespace slammer;

//
// Descriptor
//

Descriptor Descriptor::ComputeCentroid(const std::vector<const Descriptor*> descriptors) {
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

    unsigned threshold = descriptors.size() / 2 + (descriptors.size() % 2);
    Descriptor result;

    for (size_t index = 0; index < kNumBits; ++index) {
        result.descriptor_[index] = counters[index] > threshold;
    }

    return result;
}

//
// Feature matching
//

namespace {

typedef std::pair<Descriptor::Distance, size_t> Candidate;
typedef std::vector<Candidate> Candidates;

void CollectCandidates(const std::vector<Descriptor>& target,
                       const Descriptor& descriptor,
                       unsigned k,
                       Descriptor::Distance max_distance,
                       Candidates& result) {
    result.clear();

    for (size_t target_index = 0; target_index < target.size(); ++target_index) {
        auto distance = Descriptor::ComputeDistance(descriptor, target[target_index]);

        if (distance <= max_distance) {
            size_t insert_position = 0;

            while (insert_position < result.size() && result[insert_position].first <= distance) {
                ++insert_position;
            }

            if (insert_position == result.size()) {
                if (insert_position < k) {
                    result.emplace_back(distance, target_index);
                }
            } else {
                if (result.size() == k) {
                    result.pop_back();
                }

                result.insert(result.begin() + insert_position, Candidate(distance, target_index));
            }
        }
    }
}

} // namespace

Matches
slammer::ComputeMatches(const std::vector<Descriptor>& target,
                        const std::vector<Descriptor>& query,
                        Descriptor::Distance max_distance) {
    std::vector<slammer::Match> result;

    const size_t sentinel_index = target.size();
    Candidate sentinel(max_distance == std::numeric_limits<Descriptor::Distance>::max() ?
                       max_distance : max_distance + 1, sentinel_index);

    for (size_t query_index = 0; query_index < query.size(); ++query_index) {
        Candidate candidate = sentinel;

        for (size_t target_index = 0; target_index < target.size(); ++target_index) {
            auto distance = Descriptor::ComputeDistance(query[query_index], target[target_index]);
            if (distance < candidate.first) {
                candidate = Candidate(distance, target_index);
            }
        }

        if (candidate.second != sentinel_index) {
            result.emplace_back(Match {
                candidate.first, query_index, candidate.second
            });
        }
    }

    return result;
}

/// For each descriptor in the query, find up to k closest descriptors in the target that
/// is closest.
///
/// \param target       a collection of descriptors to macth against
/// \param query        a collectionn of descriptors to match
/// \param k            the number of matches to include for each descriptor
/// \param cross_check  if true, perform a reverse check matching target descriptors to query
///                     descriptors, and only inlcude matches contained in both runs
/// \param max_distance the maximum distance in order to consider a pair of matched descriptors
///                     for inclusion in the result
std::vector<Matches> 
slammer::ComputeKMatches(const std::vector<Descriptor>& target,
                         const std::vector<Descriptor>& query, 
                         unsigned k, bool cross_check,
                         Descriptor::Distance max_distance) {
    std::vector<Matches> result;

    if (!cross_check) {
        Candidates candidates;

        for (size_t query_index = 0; query_index < query.size(); ++query_index) {
            CollectCandidates(target, query[query_index], k, max_distance, candidates);

            if (!candidates.empty()) {
                Matches matches;
                std::transform(candidates.begin(), candidates.end(),
                               std::back_inserter(matches),
                               [&](const auto& candidate) {
                    return Match { candidate.first, query_index, candidate.second };
                });
                result.emplace_back(matches);
            }
        }
    } else {
        std::vector<Candidates> query_candidates, target_candidates;
        Candidates candidates;

        for (size_t query_index = 0; query_index < query.size(); ++query_index) {
            CollectCandidates(target, query[query_index], k, max_distance, candidates);
            query_candidates.push_back(candidates);
        }

        for (size_t target_index = 0; target_index < target.size(); ++target_index) {
            CollectCandidates(query, target[target_index], k, max_distance, candidates);
            target_candidates.push_back(candidates);
        }

        // Create results, filtering non-mutual matches

        Matches matches;

        for (size_t query_index = 0; query_index < query.size(); ++query_index) {
            matches.clear();

            for (const auto& candidate: query_candidates[query_index]) {
                const auto& back_references = target_candidates[candidate.second];
                if (std::find_if(back_references.begin(), back_references.end(),
                                 [&](const auto& other) {
                                       return other.second == query_index;
                                 }) != back_references.end()) {
                    matches.emplace_back(Match { candidate.first, query_index, candidate.second });
                }
            }

            if (!matches.empty()) {
                result.emplace_back(matches);
            }
        }
    }

    return result;
}
