#include "stitching.hpp"

void stitchImages(const cv::Mat& img1, const cv::Mat& img2, const cv::Mat& homography, const std::string type ){
    if(type == "simple"){
        cv::Mat panorama;
        cv::warpPerspective(img2, panorama, homography, cv::Size(img1.cols + img2.cols, img1.rows));
        cv::Mat half(panorama, cv::Rect(0, 0, img1.cols, img1.rows));
        img1.copyTo(half);

        std::string windowName = "Panorama";
        cv::namedWindow(windowName, cv::WINDOW_NORMAL);
        cv::imshow(windowName, panorama);
        cv::waitKey(0); 
        cv::destroyWindow(windowName);
    }

    else if(type == "feather"){
        cv::Mat warpedImg2;
        cv::warpPerspective(img2, warpedImg2, homography, cv::Size(img1.cols + img2.cols, img1.rows));

        cv::Mat panorama(img1.rows, img1.cols + img2.cols, img1.type());
        img1.copyTo(panorama(cv::Rect(0, 0, img1.cols, img1.rows)));

        cv::Mat gray;
        cv::cvtColor(warpedImg2, gray, cv::COLOR_BGR2GRAY);
        cv::Rect warpedRec = cv::boundingRect(gray);

        cv::Rect img1Rec(0, 0, img1.cols, img1.rows);

        cv::Rect overlap = img1Rec & warpedRec;

        cv::Mat warpedMask;
        cv::threshold(gray, warpedMask, 0, 255, cv::THRESH_BINARY);
        
        cv::Mat nonOverlapMask;
        cv::Mat img1Mask = cv::Mat::zeros(panorama.rows, panorama.cols, CV_8U);
        img1Mask(img1Rec) = 255;
        cv::bitwise_and(warpedMask, ~img1Mask, nonOverlapMask);
        warpedImg2.copyTo(panorama, nonOverlapMask);

        if (overlap.width > 0 && overlap.height > 0) {
            cv::Mat panoramaPart = panorama(overlap);
            cv::Mat warpedPart = warpedImg2(overlap);
            cv::addWeighted(panoramaPart, 0.5, warpedPart, 0.5, 0.0, panoramaPart);
        }

        std::string windowName = "Panorama (Feathered)";
        cv::namedWindow(windowName, cv::WINDOW_NORMAL);
        cv::imshow(windowName, panorama);
        cv::waitKey(0);
        cv::destroyWindow(windowName);
    }
    
}
