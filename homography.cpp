#include "homography.hpp"
#include <iostream>

cv::Mat estimateHomography(const std::vector<cv::DMatch>& matches, const KeypointsAndDescriptors& kdimg1, const KeypointsAndDescriptors& kdimg2, const int threshold, int& inlierCount){
    std::vector<cv::Point2f> points1;
    std::vector<cv::Point2f> points2;


    for (const auto& match : matches) {
        points1.push_back(kdimg1.keypoints[match.queryIdx].pt);
        points2.push_back(kdimg2.keypoints[match.trainIdx].pt);
    }
    cv::Mat mask;
    cv::Mat H = cv::findHomography(points2, points1, cv::RANSAC, threshold, mask);

    inlierCount = cv::countNonZero(mask);

    return H;
}
