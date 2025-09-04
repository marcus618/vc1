#ifndef HISTOGRAM_HPP
#define HISTOGRAM_HPP

#include <vector>
#include <string>
#include <opencv2/features2d.hpp>

// Function declaration
void plotHistograms(const std::vector<cv::DMatch>& orbMatches, const std::vector<cv::DMatch>& akazeMatches);

#endif 