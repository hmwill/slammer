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

#ifndef SLAMMER_BACKEND_H
#define SLAMMER_BACKEND_H

#pragma once

#include "slammer/slammer.h"
#include "slammer/frontend.h"
#include "slammer/map.h"
#include "slammer/keyframe_index.h"


namespace slammer {

/// Event that is fired when a keyframe pose gest adjusted
struct KeyframePoseEvent: public Event {
    KeyframePointer keyframe;
    SE3d previous_pose;
};

/// \brief This class provides the backend process for Slammer
class Backend {
public:
    using Keyframes = std::vector<KeyframePointer>;
    using Landmarks = std::vector<LandmarkPointer>;
    using Poses = std::vector<SE3d>;
    using Locations = std::vector<Point3d>;

    /// Configuration parameters for the backend process 
    struct Parameters {
        // maximum (Hamming) difference between features to be considered a match
        float max_match_distance = 12.0f;

        // minimum number of matches against previous frame to work locally
        int min_feature_matches = 50;

        // minimum number of matches against candidate frame to attempt loop closure
        int min_loop_feature_matches = 40;

        // minimum number of inliers for a good feature-based match between frames
        int min_inlier_matches = 25;

        // upper limit on the number of keyframes included in local optimization
        int max_keyframes_in_local_graph = 15;

        // number of solver iterations for local optimization
        int local_optimization_iterations = 10;

        // maximum number of search results when querying index for loop candidates
        int max_loop_candidiates = 10;

        // maximum correction distance to allow for loop closure candidate
        double max_loop_correction_distance = M_2_PI / 60.0;

        // maximum angular correction to pose orientation to allow for loop closure candidate
        double max_loop_correction_angle = std::numeric_limits<double>::max(); 

        // ICP iteration limit
        size_t max_iterations = 30;
        
        // ICP sample size
        size_t sample_size = 10;
        
        /// ICP outlier factor
        double outlier_factor = 7.16;        

        // seed value for random number generator
        int seed = 12345;
    };

    Backend(const Parameters& parameters, const Camera& rgb_camera, const Camera& depth_camera, Map& map,
            KeyframeIndex& keyframe_index);

    // Disallow copy construction and copy assignment
    Backend(const Backend&) = delete;
    Backend& operator=(const Backend&) = delete;

    /// Handler function that is called by the frontend when a new frame is available
    void HandleRgbdFrameEvent(const RgbdFrameEvent& frame);

    /// Event listeners interested in keyframe pose updates
    EventListenerList<KeyframePoseEvent> keyframe_poses;

private:
    /// map type used to temporarily map a landmark onto another one
    using LandmarkMapping = std::unordered_map<LandmarkId, LandmarkId>;

    /// Feature matches that anchor two frames relative to each other
    using FeatureMatches = std::vector<Match>;

    /// Match features from a new frame against features in a given reference frame.
    ///
    /// \param reference    Feature descriptors associated with the referene frames
    /// \param query        Feature descriptors associated with the query frame
    FeatureMatches MatchFeatures(const Descriptors& reference, const Descriptors& query);

    /// Insert a keyframe into the overall graph relative to the given refernce frame.
    ///
    /// \param keyframe         the new keyframe to insert into the graph
    /// \param reference_frame  an existing frame in the graph, relative to which the new frame
    ///                         is placed
    /// \param matches          the feature matches between the two frame used to determine the 
    ///                         relative placement   
    void ExtendGraph(const KeyframePointer& keyframe, const KeyframePointer& reference_frame,
                     const FeatureMatches& matches);

    /// Determine a loop closure for the given keyframe.
    ///
    /// \param keyframe                 the keyframe for which we want to find a loop closure 
    /// \param[out] loop_keyframe       the identified loop closure keyframe
    /// \param[out] relative_motion     initial estimate for the relative motion between the two frames
    /// \param[out] landmark_mapping    landmarks to fuse due to loop closure
    /// \return true if a loop closure candidate was identified
    bool DetermineLoopClosure(const KeyframePointer& keyframe, KeyframePointer& loop_keyframe,
                              SE3d& relative_motion, LandmarkMapping& landmark_mapping);
                                                 
    /// Starting from the specified keyframe, extract the sub-graph of keyfraems and landmarks
    /// to use for a local bundle adjustment.
    ///
    /// \param seeds        the keyframes that anchors the subgraph to be extracted
    /// \param keyframes    the keyframes to be included in the subgraph
    /// \param landmarks    the landmarks to be included in the subgraph
    /// \param subgraph_limit   maximum number of keyframes to include
    void ExtractLocalGraph(const Keyframes& seeds, Keyframes& keyframes,
                           Landmarks& landmarks, size_t subgraph_limit);

    /// Optimize poses within the subgraph induced by the given keyframes for loop closure.
    /// Rather than updating pose information in place, we return it as a new vector of 
    /// coordinates that are aligned based on index.
    ///
    /// \param keyframes    the keyframes included in the subgraph
    /// \param from         origin of the new loop edge to be introduced   
    /// \param to           destination of the new loop edge to be introduced
    /// \param relative_motion an estimate for the relative motion between the poses to connect 
    /// \return             the new keyframe poses calculated as result of the optimization process
    Poses OptimizeLoopPoses(const Keyframes& keyframes, const KeyframePointer& from, 
                            const KeyframePointer& to, SE3d relative_motion);

    /// Estimate locations for the landmarks based on a new set of pose estimates for a given set
    /// of keyframes
    ///
    /// \param keyframes    the keyframes included in the subgraph
    /// \param landmarks    the landmarks included in the subgraph
    /// \param poses        the new keyframe poses calculated as result of the optimization process
    /// \return             an (initial) estimate of the landmark location based on the graph structure 
    ///                     and keyframe poses
    Locations EstimateLocations(const Keyframes& keyframes, const Landmarks& landmarks,
                                const Poses& poses);

    /// Optimize poses and locations within the subgraph induced by the given keyframes and landmarks.
    /// Rather than updating pose and location information in place, we collect them into new
    /// data structures that are aligned based on index.
    ///
    /// Landmarks in the sub-graph that are in the range of the provided mapping will be treated as identical
    /// to the landmark they are mapped to.
    ///
    /// \param keyframes    the keyframes included in the subgraph
    /// \param landmarks    the landmarks included in the subgraph
    /// \param poses        (out) the new keyframe poses calculated as result of the optimization process
    /// \param locations    (out) the new landmark locations calculated as result of the optimization process
    /// \param mapping      assume the unification of landmarks based on this mapping    
    ///\param poses_as_input utilize poses and locations both as input and output value
    void OptimizePosesAndLocations(const Keyframes& keyframes, const Landmarks& landmarks,
                                   Poses& poses, Locations& locations, 
                                   const LandmarkMapping& mapping = LandmarkMapping {},
                                   bool poses_as_input = false);

    /// Incorporate updated keyframe pose and landmark location information into the map.
    ///
    /// \param keyframes    the keyframes included in the subgraph
    /// \param landmarks    the landmarks included in the subgraph
    /// \param poses        the new keyframe poses to incorporate
    /// \param locations    the new landmark locations to incorporate
    void UpdatePosesAndLocations(const Keyframes& keyframes, const Landmarks& landmarks,
                                 const Poses& poses, const Locations& locations);

private:
    /// Configuration parameters
    Parameters parameters_;

    /// Parameters describing the RGB camera
    const Camera& rgb_camera_;

    /// Parameters describing the depth camera
    const Camera& depth_camera_;
    
    /// The sparse map we are populating
    Map& map_;

    /// The keyframe index
    KeyframeIndex& keyframe_index_;

    /// Random number generator to use
    std::default_random_engine random_engine_;
};

} // namespace slammer

#endif //ndef SLAMMER_BACKEND_H