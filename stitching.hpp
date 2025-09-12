#ifndef STITCHING_HPP
#define STITCHING_HPP

#include <opencv2/opencv.hpp>

void stitchImages(const cv::Mat& img1, const cv::Mat& img2, const cv::Mat& homography, const std::string type );

#endif