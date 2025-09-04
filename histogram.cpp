#include "histogram.hpp"
#include <iostream>
#include <fstream>
#include <filesystem> 

#define RESULTS_DIR "DistancesForHistogram/"

void plotHistograms(const std::vector<cv::DMatch>& orbMatches, const std::vector<cv::DMatch>& akazeMatches) {
    if (orbMatches.empty() || akazeMatches.empty()) {
        return;
    }

    std::filesystem::create_directory(RESULTS_DIR);

    //for orb
    std::ofstream orbfile(std::string(RESULTS_DIR) + "orbDistances.txt");
    
    orbfile << "distance\n";
    for (const auto& match : orbMatches) {
        orbfile << match.distance << "\n";
    }
    orbfile.close();

    //for akaze
    std::ofstream akazefile(std::string(RESULTS_DIR) + "akazeDistances.txt");
    akazefile << "distance\n";
    for (const auto& match : akazeMatches) {
        akazefile << match.distance << "\n";
    }
    akazefile.close();
}