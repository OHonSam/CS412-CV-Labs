#pragma once
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include "HarrisDetector.hpp"
#include "DoGDetector.hpp"
#include "BlobDetector.hpp"
#include "SIFTDescriptor.hpp"
#include "LBPDescriptor.hpp"


struct MatchContext {
    cv::Mat img1, img2, gray1, gray2;
    IKeypointDetector* detector = nullptr;
    IDescriptor* descriptor = nullptr;

    int ratioTest_x100 = 75;
};


// Matching functions
void matchFeatures(const std::string& detectorType, const std::string& descriptorType, 
                  const std::string& img1Path, const std::string& img2Path);
void computeSIFTDescriptors(const cv::Mat& gray, 
        const std::vector<cv::KeyPoint>& keypoints, 
        cv::Mat& descriptors);
void computeLBPDescriptors(const cv::Mat& gray, 
        const std::vector<cv::KeyPoint>& keypoints, 
        cv::Mat& descriptors);