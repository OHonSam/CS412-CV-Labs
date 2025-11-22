#include "FeatureMatching.hpp"
#include <iostream>

void matchFeatures(const std::string& detectorType, const std::string& descriptorType, 
                  const std::string& img1Path, const std::string& img2Path) {
    std::cout << "Matching " << img1Path << " and " << img2Path << std::endl;
    std::cout << "Detector: " << detectorType << ", Descriptor: " << descriptorType << std::endl;
    if (detectorType != "harris" && detectorType != "dog" && detectorType != "blob") {
        std::cerr << "Error: Unknown detector type '" << detectorType << "'" << std::endl;
        return;
    }

    if (descriptorType != "sift" && descriptorType != "lbp") {
        std::cerr << "Error: Unknown descriptor type '" << descriptorType << "'" << std::endl;
        return;
    }

    // Load images
    cv::Mat img1 = cv::imread(img1Path, cv::IMREAD_COLOR);
    cv::Mat img2 = cv::imread(img2Path, cv::IMREAD_COLOR);
    if (img1.empty() || img2.empty()) {
        std::cerr << "Error: Could not read one of the images." << std::endl;
        return;
    }

    cv::Mat gray1, gray2;
    cv::cvtColor(img1, gray1, cv::COLOR_BGR2GRAY);
    cv::cvtColor(img2, gray2, cv::COLOR_BGR2GRAY);

    // // Detect keypoints
    // std::vector<cv::KeyPoint> keypoints1, keypoints2;
    // if (detectorType == "harris") {
    //     HarrisContext ctx1, ctx2;
    //     ctx1.src = img1;
    //     ctx2.src = img2;

    //     ctx1.gray = gray1;
    //     ctx2.gray = gray2;
    // } else if (detectorType == "dog") {
    //     DoGContext ctx1, ctx2;
    //     ctx1.src = img1;
    //     ctx2.src = img2;

    //     ctx1.gray = gray1;
    //     ctx2.gray = gray2;
    // } else if (detectorType == "blob") {
    //     BlobContext ctx1, ctx2;
    //     ctx1.src = img1;
    //     ctx2.src = img2;

    //     ctx1.gray = gray1;
    //     ctx2.gray = gray2;
    // }
}