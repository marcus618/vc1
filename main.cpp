#include <opencv2/opencv.hpp>
#include <iostream>
#include <chrono>
#include <opencv2/core/utils/logger.hpp>
#include <fstream>
#include "histogram.hpp"


struct KeypointsAndDescriptors {
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;
};

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
    //Load the original images
    cv::Mat img1 = cv::imread(path1, cv::IMREAD_COLOR);
    cv::Mat img2 = cv::imread(path2, cv::IMREAD_COLOR);

    //Create an image to draw the matches on
    cv::Mat img_matches;
    cv::drawMatches(img1, res1.keypoints, img2, res2.keypoints, matches, img_matches);

    //Create a window and display the matches
    std::string windowTitle = "Matches (" + detectorName + ")";
    cv::namedWindow(windowTitle, cv::WINDOW_NORMAL);
    cv::imshow(windowTitle, img_matches);
    cv::waitKey(0);
    cv::destroyWindow(windowTitle);

	return matches;
}

int main() {
	    cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_WARNING);

   	 	std::vector<std::string> imageSetDirs = {"imagesets/imageset1", "imagesets/imageset2", "imagesets/imageset3"};
		std::vector<std::vector<KeypointsAndDescriptors>> allOrbResults;
        std::vector<std::vector<KeypointsAndDescriptors>> allAkazeResults;
		std::vector<std::vector<std::string>> allImagePaths;

		//feature detection
		for(const auto& dir:  imageSetDirs){
			std::vector<cv::String> imagePaths;
			cv::glob(dir + "/*.jpg", imagePaths, false);

			std::vector<KeypointsAndDescriptors> currentSetOrbResults;
            std::vector<KeypointsAndDescriptors> currentSetAkazeResults;
			std::vector<std::string> currentSetImagePaths;

			for(const auto& imagePath: imagePaths){
				auto image = cv::imread(imagePath, cv::IMREAD_COLOR);

				if(image.empty()){
            		std::cout << "Could not read the image: " << imagePath << std::endl;
					continue;
        		}
				
				currentSetImagePaths.push_back(imagePath);

                KeypointsAndDescriptors orbRes = featureDetection("ORB", image);
                currentSetOrbResults.push_back(orbRes);

                KeypointsAndDescriptors akazeRes = featureDetection("AKAZE", image);
                currentSetAkazeResults.push_back(akazeRes);

			}
			allOrbResults.push_back(currentSetOrbResults);
            allAkazeResults.push_back(currentSetAkazeResults);
			allImagePaths.push_back(currentSetImagePaths);
		}

		//using feature detection for matching
		for (size_t i = 0; i < imageSetDirs.size(); ++i){
			std::vector<cv::DMatch> totalOrbMatches;
            std::vector<cv::DMatch> totalAkazeMatches;
			for(size_t j = 0; j < allImagePaths[i].size(); ++j){
				for(size_t k = j + 1; k < allImagePaths[i].size(); ++k){
					auto orbMatches = match("ORB", allImagePaths[i][j], allOrbResults[i][j], allImagePaths[i][k], allOrbResults[i][k]);
					totalOrbMatches.insert(totalOrbMatches.end(), orbMatches.begin(), orbMatches.end());

					auto akazeMatches = match("AKAZE", allImagePaths[i][j], allAkazeResults[i][j], allImagePaths[i][k], allAkazeResults[i][k]);
					totalAkazeMatches.insert(totalAkazeMatches.end(), akazeMatches.begin(), akazeMatches.end());
				}
			}
			//use matching to plot histograms
			plotHistograms(totalOrbMatches, totalAkazeMatches);
		}
		


		

        return 0;

}


