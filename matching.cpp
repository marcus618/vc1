#include "matching.hpp"

std::vector<cv::DMatch> match(const std::string& detectorName, const std::string& path1, const KeypointsAndDescriptors& res1, const std::string& path2, const KeypointsAndDescriptors& res2){
	int normType = (detectorName == "ORB") ? cv::NORM_HAMMING : cv::NORM_L2;
	auto bf = cv::BFMatcher::create(normType);

	std::vector<cv::DMatch> matches;
	
    auto t1 = std::chrono::high_resolution_clock::now();
    bf->match(res1.descriptors, res2.descriptors, matches);
	auto t2 = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();

	std::cout << "Matches with " << detectorName << " between " 
              << path1.substr(path1.find_last_of("/\\") + 1) << " and " 
              << path2.substr(path2.find_last_of("/\\") + 1) << ": " 
              << matches.size() << " in " << duration << " ms" << std::endl;



	//Visualization Code
    cv::Mat img1 = cv::imread(path1, cv::IMREAD_COLOR);
    cv::Mat img2 = cv::imread(path2, cv::IMREAD_COLOR);

    cv::Mat img_matches;
    cv::drawMatches(img1, res1.keypoints, img2, res2.keypoints, matches, img_matches);

    std::string windowTitle = "Matches (" + detectorName + ")";
    cv::namedWindow(windowTitle, cv::WINDOW_NORMAL);
    cv::imshow(windowTitle, img_matches);
    cv::waitKey(0);
    cv::destroyWindow(windowTitle);

	return matches;
}