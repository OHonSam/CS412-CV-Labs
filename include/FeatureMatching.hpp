#pragma once
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include "HarrisDetector.hpp"
#include "DoGDetector.hpp"
#include "BlobDetector.hpp"



struct MatchContext {
    cv::Mat img1;
    cv::Mat img2;
    cv::Mat gray1;
    cv::Mat gray2;

    std::string detectorType;
    std::string descriptorType;

    std::vector<cv::KeyPoint> keypoints1;
    std::vector<cv::KeyPoint> keypoints2;
    cv::Mat descriptors1;
    cv::Mat descriptors2;
    std::vector<cv::DMatch> goodMatches;

    HarrisContext harrisContext;
    DoGContext dogContext;
    BlobContext blobContext;

    int ratioThreshold = 75; // Lowe's ratio test threshold (scaled by 100)
};

// Matching functions
void onMatchTrackbar(int, void* userData);
void matchFeatures(const std::string& detectorType, const std::string& descriptorType, 
                  const std::string& img1Path, const std::string& img2Path);
void matchFeaturesCamera(const std::string& detectorType, const std::string& descriptorType);
void computeSIFTDescriptors(const cv::Mat& gray, 
        const std::vector<cv::KeyPoint>& keypoints, 
        cv::Mat& descriptors);
void computeLBPDescriptors(const cv::Mat& gray, 
        const std::vector<cv::KeyPoint>& keypoints, 
        cv::Mat& descriptors);