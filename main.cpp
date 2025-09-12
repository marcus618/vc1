#include <opencv2/opencv.hpp>
#include <iostream>
#include <chrono>
#include <opencv2/core/utils/logger.hpp>
#include <fstream>
#include "histogram.hpp"
#include "featureDetection.hpp"
#include "matching.hpp"
#include "homography.hpp"
#include "stitching.hpp"




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


		std::vector<int> thresholds = {1,5,10};
		std::vector<cv::DMatch> allOrbMatches;
        std::vector<cv::DMatch> allAkazeMatches;
        for (size_t i = 0; i < imageSetDirs.size(); ++i){

            //match
            auto orbMatches = match("ORB", allImagePaths[i][0], allOrbResults[i][0], allImagePaths[i][1], allOrbResults[i][1]);
            auto akazeMatches = match("AKAZE", allImagePaths[i][0], allAkazeResults[i][0], allImagePaths[i][1], allAkazeResults[i][1]);

			//keypoint Count vs. match quality
            size_t totalAkazeKeypointsForSet = allAkazeResults[i][0].keypoints.size() + allAkazeResults[i][1].keypoints.size();
            double totalAkazeDistance = 0.0;
            for(const auto& match : akazeMatches) {
                totalAkazeDistance += match.distance;
            }
            double avgAkazeDistance =  totalAkazeDistance / akazeMatches.size();

            std::cout << "\n--- Analysis for " << imageSetDirs[i] << " ---" << std::endl;
            std::cout << "AKAZE Keypoint Count: " << totalAkazeKeypointsForSet 
                      << ", Average Match Distance: " << avgAkazeDistance << std::endl;

			allOrbMatches.insert(allOrbMatches.end(), orbMatches.begin(), orbMatches.end());
            allAkazeMatches.insert(allAkazeMatches.end(), akazeMatches.begin(), akazeMatches.end());

            //homography estimation with different thresholds 
            for (int t : thresholds) {
                int inlierCountORB = 0;
				int inlierCountAKAZE = 0;

                auto t1_orb = std::chrono::high_resolution_clock::now();
                cv::Mat H_orb = estimateHomography(orbMatches, allOrbResults[i][0], allOrbResults[i][1], t, inlierCountORB);
                auto t2_orb = std::chrono::high_resolution_clock::now();
                auto duration_orb = std::chrono::duration_cast<std::chrono::milliseconds>(t2_orb - t1_orb).count();
                size_t totalOrbKeypoints = allOrbResults[i][0].keypoints.size() + allOrbResults[i][1].keypoints.size();
                float orbInlierRatio = orbMatches.empty() ? 0.0f : static_cast<float>(inlierCountORB) / orbMatches.size();
                std::cout << "ORB (threshold: " << t << ") - Keypoints: " << totalOrbKeypoints << ", Inlier Ratio: " << orbInlierRatio << " (" << inlierCountORB << "/" << orbMatches.size() << "), Time: " << duration_orb << " ms" << std::endl;

				auto t1_akaze = std::chrono::high_resolution_clock::now();
                cv::Mat H_akaze = estimateHomography(akazeMatches, allAkazeResults[i][0], allAkazeResults[i][1], t, inlierCountAKAZE);
                auto t2_akaze = std::chrono::high_resolution_clock::now();
                auto duration_akaze = std::chrono::duration_cast<std::chrono::milliseconds>(t2_akaze - t1_akaze).count();
                size_t totalAkazeKeypoints = allAkazeResults[i][0].keypoints.size() + allAkazeResults[i][1].keypoints.size();
                float akazeInlierRatio = akazeMatches.empty() ? 0.0f : static_cast<float>(inlierCountAKAZE) / akazeMatches.size();
                std::cout << "AKAZE (threshold: " << t << ") - Keypoints: " << totalAkazeKeypoints << ", Inlier Ratio: " << akazeInlierRatio << " (" << inlierCountAKAZE << "/" << akazeMatches.size() << "), Time: " << duration_akaze << " ms" << std::endl;

				cv::Mat img1 = cv::imread(allImagePaths[i][0], cv::IMREAD_COLOR);
				cv::Mat img2 = cv::imread(allImagePaths[i][1], cv::IMREAD_COLOR);

                if (!H_orb.empty()) {
                    stitchImages(img1, img2, H_orb, "simple");
					stitchImages(img1, img2, H_orb, "feather");
                }
				if (!H_akaze.empty()){
					stitchImages(img1, img2, H_akaze, "simple");
					stitchImages(img1, img2, H_akaze, "feather");
				}
			}
			cv::destroyAllWindows();

        }
		plotHistograms(allOrbMatches, allAkazeMatches);
        
        return 0;

}
		



