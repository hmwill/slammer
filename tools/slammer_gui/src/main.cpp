// BSD 3-Clause License
//
// Copyright (c) 2021, 2022, Hans-Martin Will
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

#include <cstdlib>

#include "slammer/slammer.h"
#include "slammer/loris/opencv_utils.h"

#include "opencv2/opencv.hpp"
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/viz.hpp"

using namespace slammer;
using namespace slammer::loris;

const std::string kDataSet { "data/cafe1-1"s };
const std::string kImage1 { "data/cafe1-1/color/1560004885.446172.png"s };
// const std::string kImage1 { "data/images/soup1.jpg"s };
//const std::string kImage2 { "data/cafe1-1/color/1560004885.679670.png"s };
// const std::string kImage2 { "data/images/soup2.jpg"s };
const std::string kImage2 { "data/cafe1-1/color/1560004888.047959.png"s };

Result<cv::Matx33d> GetCameraIntrinsicMatrix() {
    auto maybe_info = ReadSensorInfo(kDataSet);

    if (maybe_info.failed()) {
        return maybe_info.error();
    }

    const auto& intrinsics = maybe_info.value().d400_color_optical_frame.intrinsics;
    auto result = cv::Matx33d::zeros();
    result(0, 0) = intrinsics.at<double>(0);
    result(0, 2) = intrinsics.at<double>(1);
    result(1, 1) = intrinsics.at<double>(2);
    result(1, 2) = intrinsics.at<double>(3);
    result(2, 2) = 1.0;

    return result;
}

Result<cv::Mat> GetImage(const std::string& path) {
    auto image = cv::imread(path, cv::IMREAD_UNCHANGED);

    if (image.empty()) {
            std::string error_message = "Could not read image file " + path;
            return Error(error_message);
    } else {
        return image;
    }
}

// triangulate using Linear LS-Method
cv::Vec3d triangulate(const cv::Mat &p1, const cv::Mat &p2, const cv::Vec2d &u1, const cv::Vec2d &u2) {

	// system of equations assuming image=[u,v] and X=[x,y,z,1]
	// from u(p3.X)= p1.X and v(p3.X)=p2.X
	cv::Matx43d A(u1(0)*p1.at<double>(2, 0) - p1.at<double>(0, 0), u1(0)*p1.at<double>(2, 1) - p1.at<double>(0, 1), u1(0)*p1.at<double>(2, 2) - p1.at<double>(0, 2),
		u1(1)*p1.at<double>(2, 0) - p1.at<double>(1, 0), u1(1)*p1.at<double>(2, 1) - p1.at<double>(1, 1), u1(1)*p1.at<double>(2, 2) - p1.at<double>(1, 2),
		u2(0)*p2.at<double>(2, 0) - p2.at<double>(0, 0), u2(0)*p2.at<double>(2, 1) - p2.at<double>(0, 1), u2(0)*p2.at<double>(2, 2) - p2.at<double>(0, 2),
		u2(1)*p2.at<double>(2, 0) - p2.at<double>(1, 0), u2(1)*p2.at<double>(2, 1) - p2.at<double>(1, 1), u2(1)*p2.at<double>(2, 2) - p2.at<double>(1, 2));

	cv::Matx41d B(p1.at<double>(0, 3) - u1(0)*p1.at<double>(2, 3),
		          p1.at<double>(1, 3) - u1(1)*p1.at<double>(2, 3),
		          p2.at<double>(0, 3) - u2(0)*p2.at<double>(2, 3),
		          p2.at<double>(1, 3) - u2(1)*p2.at<double>(2, 3));

	// X contains the 3D coordinate of the reconstructed point
	cv::Vec3d X;

	// solve AX=B
	cv::solve(A, B, X, cv::DECOMP_SVD);

	return X;
}

// triangulate a vector of image points
void triangulate(const cv::Mat &p1, const cv::Mat &p2, const std::vector<cv::Vec2d> &pts1, const std::vector<cv::Vec2d> &pts2, std::vector<cv::Vec3d> &pts3D) {

	for (int i = 0; i < pts1.size(); i++) {
		pts3D.push_back(triangulate(p1, p2, pts1[i], pts2[i]));
	}
}

/*
First excercise:

- Load two images
- Identify features in both images
- Calculate their descriptors
- Match them
- Determine the implied camera transformation
- Visualize the two images and the 3D point cloud of the matched features

Second exercise:

- Expand the above with bundle adjustement

Third exercise:

- Expand the above with more than two pictures

*/

int main(int argc, char *argv) {
    cv::Mat image1 = GetImage(kImage1).value();
    cv::Mat image2 = GetImage(kImage2).value();
    cv::Matx33d camera_matrix = GetCameraIntrinsicMatrix().value();
    // cv::Matx33d camera_matrix(1286.540148375528, 0.0, 929.5596785987797,
    //     0.0, 1272.8889372475517, 586.0340979684613,
    //     0.0, 0.0, 1.0);

    cv::Mat cameraDistCoeffs = (cv::Mat1d(1, 5) << 0.0, 0.0, 0.0, 0.0, 0.0);

	// Read the camera calibration parameters
	// cv::Mat camera_matrix;
	//cv::Mat cameraDistCoeffs;
	// cv::FileStorage fs("data/images/calib.xml", cv::FileStorage::READ);
	// fs["Intrinsic"] >> camera_matrix;
	//fs["Distortion"] >> cameraDistCoeffs;

	// camera_matrix.at<double>(0, 2) = 268.;
	// camera_matrix.at<double>(1, 2) = 178;

    std::cout << "Camera matrix: " << camera_matrix << std::endl;
	cv::Matx33f cMatrix(camera_matrix);
	
	// undistort images
	// cv::Mat image1;
	// cv::Mat image2;

	// cv::undistort(image1In, image1, camera_matrix, cameraDistCoeffs);
	// cv::undistort(image2In, image2, camera_matrix, cameraDistCoeffs);

    // identify keypoints
    std::vector<cv::KeyPoint> keypoints1;
    std::vector<cv::KeyPoint> keypoints2;
    cv::Mat descriptors1, descriptors2;

    cv::Ptr<cv::Feature2D> ptr_features = cv::ORB::create();
    ptr_features->detectAndCompute(image1, cv::noArray(), keypoints1, descriptors1);
    ptr_features->detectAndCompute(image2, cv::noArray(), keypoints2, descriptors2);

	std::cout << "Num rows: " << descriptors1.rows << std::endl;
	std::cout << "Num cols: " << descriptors1.cols << std::endl;
	std::cout << "Type: " << descriptors1.type() << std::endl;

	//std::cout << "Descriptor1:" << std::endl << descriptors1 << std::endl;

    cv::BFMatcher matcher(cv::NORM_HAMMING, true);
    std::vector<cv::DMatch> matches;
    matcher.match(descriptors1, descriptors2, matches);

	std::cout << "Number of matches: " << matches.size() << std::endl;;

	// draw the matches
	cv::Mat imageMatches;
	cv::drawMatches(image1, keypoints1,  // 1st image and its keypoints
		image2, keypoints2,  // 2nd image and its keypoints
		matches,			// the matches
		imageMatches,		// the image produced
		cv::Scalar(255, 255, 255),  // color of the lines
		cv::Scalar(255, 255, 255),  // color of the keypoints
		std::vector<char>(),
		cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
	cv::namedWindow("Matches");
	cv::imshow("Matches", imageMatches);

    std::vector<cv::Point2f> points1, points2;

    for (const auto& elem: matches) {
        float x = keypoints1[elem.queryIdx].pt.x;
        float y = keypoints1[elem.queryIdx].pt.y;
        points1.push_back(cv::Point2f(x, y));

        x = keypoints2[elem.trainIdx].pt.x;
        y = keypoints2[elem.trainIdx].pt.y;
        points2.push_back(cv::Point2f(x, y));
    }
    
    cv::Mat inliers;
    cv::Mat essential = 
        cv::findEssentialMat(points2, points1, camera_matrix, cv::RANSAC, 0.90, 1.0, inliers);

	std::cout << "Essential = " << essential << std::endl;

	// Analyze number of inliers and their errors;
	int num_inliers = 0;
	double sum_squared = 0.0;
	double max_error = 0.0;

	cv::Mat inv_camera_matrix = cv::Mat(camera_matrix).inv();
	cv::Mat fundamental_matrix = inv_camera_matrix.t() * essential * inv_camera_matrix;

	std::cout << "Fundamental = " << fundamental_matrix << std::endl;

	for (int i = 0; i < inliers.rows; i++) {

		if (inliers.at<uchar>(i)) {
			++num_inliers;
			cv::Mat left = (cv::Mat1d(1, 3) << points1[i].x, points1[i].y, 1.0);
			cv::Mat right = (cv::Mat1d(3, 1) << points2[i].x, points2[i].y, 1.0);
			cv::Mat line = fundamental_matrix * right;
			auto factor = 1.0/sqrt(line.at<double>(0,0) * line.at<double>(0,0) + line.at<double>(1,0) * line.at<double>(1,0));
			line = line * factor;
			cv::Mat value_mat = (left * line);
			double value = value_mat.at<double>(0, 0);
			// std::cout << "Line<" << i << "> = " << line << std::endl;
			// std::cout << "Left<" << i << "> = " << left << std::endl;
			// std::cout << "Right<" << i << "> = " << right << std::endl;
			// std::cout << "Product<" << i << "> = " << value << std::endl;

			if (fabs(value) > max_error) {
				max_error = fabs(value);
			}

			sum_squared += value * value;
		}
	}

	std::cout << "Number of inliers: " << num_inliers << std::endl;
	std::cout << "Standard deviation: " << sqrt(sum_squared / num_inliers) << std::endl;
	std::cout << "Max error: " << max_error << std::endl;

	// draw the inlier matches
	cv::drawMatches(image1, keypoints1,  	// 1st image and its keypoints
		image2, keypoints2,  				// 2nd image and its keypoints
		matches,							// the matches
		imageMatches,						// the image produced
		cv::Scalar(255, 255, 255),  		// color of the lines
		cv::Scalar(255, 255, 255),  		// color of the keypoints
		inliers,
		cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
	cv::namedWindow("Inliers matches");
	cv::imshow("Inliers matches", imageMatches);

    cv::Mat rotation, translation;
    int passing = cv::recoverPose(essential, points2, points1, camera_matrix, rotation, translation, inliers);
	std::cout << "Number points passing after recover pose: " << passing << std::endl;

	std::cout << "Rotation =" << rotation << std::endl;
	std::cout << "Translation =" << translation << std::endl;

	translation = translation * 0.1;

	double angle = acos(( rotation.at<double>(0, 0) + rotation.at<double>(1, 1) + rotation.at<double>(2, 2) - 1)/2);
	std::cout << "Angle = " << angle << " Degrees = " << angle * (360.0 / M_PI) << std::endl;

	// compose projection matrix from R,T 
	cv::Mat projection2(3, 4, CV_64F);        // the 3x4 projection matrix
	rotation.copyTo(projection2(cv::Rect(0, 0, 3, 3)));
	translation.copyTo(projection2.colRange(3, 4));

	// compose generic projection matrix 
	cv::Mat projection1(3, 4, CV_64F, 0.);    // the 3x4 projection matrix
	cv::Mat diag(cv::Mat::eye(3, 3, CV_64F));
	diag.copyTo(projection1(cv::Rect(0, 0, 3, 3)));

	std::cout << "First Projection matrix=" << projection1 << std::endl;
	std::cout << "Second Projection matrix=" << projection2 << std::endl;

	// to contain the inliers
	std::vector<cv::Vec2d> inlierPts1;
	std::vector<cv::Vec2d> inlierPts2;

	// create inliers input point vector for triangulation
	int j(0); 
	for (int i = 0; i < inliers.rows; i++) {

		if (inliers.at<uchar>(i)) {
			inlierPts1.push_back(cv::Vec2d(points1[i].x, points1[i].y));
			inlierPts2.push_back(cv::Vec2d(points2[i].x, points2[i].y));
		}
	}

	// undistort and normalize the image points
	std::vector<cv::Vec2d> points1u;
	cv::undistortPoints(inlierPts1, points1u, camera_matrix, cameraDistCoeffs);
	std::vector<cv::Vec2d> points2u;
	cv::undistortPoints(inlierPts2, points2u, camera_matrix, cameraDistCoeffs);

	// triangulation
	std::vector<cv::Vec3d> points3D;
	triangulate(projection2, projection1, points1u, points2u, points3D);

	// choose one point for visualization
    const int kPoint = 12;
	// cv::Vec3d testPoint = triangulate(projection1, projection2, points1u[kPoint], points2u[kPoint]);
	cv::Vec3d testPoint = points3D[kPoint];
	cv::viz::WSphere point3D(testPoint, 0.05, 10, cv::viz::Color::red());
	// its associated line of projection
	double lenght(4.);
	cv::viz::WLine line1(cv::Point3d(0., 0., 0.), cv::Point3d(lenght*points1u[kPoint](0), lenght*points1u[kPoint](1), lenght), cv::viz::Color::green());
	cv::viz::WLine line2(cv::Point3d(0., 0., 0.), cv::Point3d(lenght*points2u[kPoint](0), lenght*points2u[kPoint](1), lenght), cv::viz::Color::green());

	// the reconstructed cloud of 3D points
	cv::viz::WCloud cloud(points3D, cv::viz::Color::blue());
	cloud.setRenderingProperty(cv::viz::POINT_SIZE, 3.);
    
    cv::viz::Viz3d visualizer("Viz Window");
    visualizer.setBackgroundColor(cv::viz::Color::white());

    cv::viz::WCameraPosition cam(cMatrix, image1, 1.0, cv::viz::Color::black());
    cv::viz::WCameraPosition cam2(cMatrix, image2, 1.0, cv::viz::Color::black());
    visualizer.showWidget("Camera 1", cam);
    visualizer.showWidget("Camera 2", cam2);
	visualizer.showWidget("Cloud", cloud);
	visualizer.showWidget("Line1", line1);
	visualizer.showWidget("Line2", line2);
	visualizer.showWidget("Triangulated", point3D);

	cv::Affine3d pose(rotation, translation);
    visualizer.setWidgetPose("Camera 2", pose);
	visualizer.setWidgetPose("Line2", pose);

    while (!visualizer.wasStopped()) {
        visualizer.spinOnce(/*1, true*/);
    }

    return EXIT_SUCCESS;
}