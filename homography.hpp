#ifndef HOMOGRAPHY_HPP
#define HOMOGRAPHY_HPP

#include <opencv2/opencv.hpp>
#include <vector>
#include "featureDetection.hpp" 

// CORRECTED: Added a comma between 'matches' and 'kdimg1'
cv::Mat estimateHomography(const std::vector<cv::DMatch>& matches, const KeypointsAndDescriptors& kdimg1, const KeypointsAndDescriptors& kdimg2, const int threshold, int& inlierCount);

#endif // HOMOGRAPHY_HPP