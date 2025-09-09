#include "featureDetection.hpp"
#include <iostream>


KeypointsAndDescriptors featureDetection(const std::string& detectorName,  cv::Mat& image) {
	std::vector<cv::KeyPoint> keypoints;
	cv::Mat descriptors;


	if(detectorName == "ORB"){
		auto orb = cv::ORB::create(1000);
		orb->detectAndCompute(image, cv::noArray(), keypoints, descriptors);
		std::cout << "Keypoint count with Orb: " << keypoints.size() << std::endl;
		KeypointsAndDescriptors keypointsAndDescriptors = {keypoints, descriptors};
		return keypointsAndDescriptors;
	}
	else if(detectorName == "AKAZE"){
		cv::Ptr<cv::AKAZE> akaze = cv::AKAZE::create();
		akaze->detectAndCompute(image, cv::noArray(), keypoints, descriptors);
		std::cout << "Keypoint count with Akaze: " << keypoints.size() << std::endl;
		KeypointsAndDescriptors keypointsAndDescriptors = {keypoints, descriptors};
		return keypointsAndDescriptors;
	}
	
}