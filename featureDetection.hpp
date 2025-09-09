#ifndef FEATURE_DETECTION_HPP
#define FEATURE_DETECTION_HPP

#include <opencv2/opencv.hpp>
#include <vector>
#include <string>


struct KeypointsAndDescriptors {
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;
};

KeypointsAndDescriptors featureDetection(const std::string& detectorName, cv::Mat& image);

#endif // FEATURE_DETECTION_HPP