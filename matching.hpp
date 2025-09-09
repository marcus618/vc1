#ifndef MATCHING_HPP
#define MATCHING_HPP

#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include "featureDetection.hpp" 
#include <iostream>
#include <chrono>

std::vector<cv::DMatch> match(const std::string& detectorName, const std::string& path1, const KeypointsAndDescriptors& res1, const std::string& path2, const KeypointsAndDescriptors& res2);

#endif // MATCHING_HPP