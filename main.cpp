#include <opencv2/opencv.hpp>
#include <iostream>
#include <chrono>
#include <opencv2/core/utils/logger.hpp>
#include <fstream>
#include "histogram.hpp"
#include "featureDetection.hpp"
#include "matching.hpp"
#include "homography.hpp"



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
				auto image = cv::imread(imagePath, cv::IMREAD_GRAYSCALE);
				
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


		std::vector<int> thresholds = {2, 4, 6, 8, 10};
        for (size_t i = 0; i < imageSetDirs.size(); ++i){

            //match
            auto orbMatches = match("ORB", allImagePaths[i][0], allOrbResults[i][0], allImagePaths[i][1], allOrbResults[i][1]);
            auto akazeMatches = match("AKAZE", allImagePaths[i][0], allAkazeResults[i][0], allImagePaths[i][1], allAkazeResults[i][1]);

            //homography estimation with different thresholds 
            cv::Mat best_H;
            int maxInliers = -1;

            for (int t : thresholds) {
                int inlierCountORB = 0;
                cv::Mat H_orb = estimateHomography(orbMatches, allOrbResults[i][0], allOrbResults[i][1], t, inlierCountORB);
                std::cout << "ORB threshold: " << t << " Inliers: " << inlierCountORB << "/" << orbMatches.size() << std::endl;
                if (!H_orb.empty() && inlierCountORB > maxInliers) {
                    maxInliers = inlierCountORB;
                    best_H = H_orb;
                }

                int inlierCountAKAZE = 0;
                cv::Mat H_akaze = estimateHomography(akazeMatches, allAkazeResults[i][0], allAkazeResults[i][1], t, inlierCountAKAZE);
                std::cout << "AKAZE threshold: " << t << " Inliers: " << inlierCountAKAZE << "/" << akazeMatches.size() << std::endl;
                if (!H_akaze.empty() && inlierCountAKAZE > maxInliers) {
                    maxInliers = inlierCountAKAZE;
                    best_H = H_akaze;
                }
            }

            if (best_H.empty()) {
                std::cerr << "Homography could not be estimated!" << std::endl;
                continue;
            }
            std::cout << "Using homography with inliers: " << maxInliers << std::endl;

            //stitch images using the best homography
            cv::Mat img1 = cv::imread(allImagePaths[i][0], cv::IMREAD_GRAYSCALE);
            cv::Mat img2 = cv::imread(allImagePaths[i][1], cv::IMREAD_GRAYSCALE);

            cv::Mat panorama;
            cv::warpPerspective(img2, panorama, best_H, cv::Size(img1.cols + img2.cols, img1.rows));

            cv::Mat half(panorama, cv::Rect(0, 0, img1.cols, img1.rows));
            img1.copyTo(half);
			
			std::string windowName = "Panorama " + std::to_string(i);
            cv::namedWindow(windowName, cv::WINDOW_NORMAL);
            cv::imshow(windowName, panorama);
			cv::waitKey(0); // Wait for a key press
            cv::destroyWindow(windowName);

            
            plotHistograms(orbMatches, akazeMatches);
            


        }
        
        cv::waitKey(0); // Wait for a key press before closing windows

        return 0;

}
		



