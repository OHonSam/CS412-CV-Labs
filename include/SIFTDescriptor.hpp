#pragma once
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

struct SIFTContext {
        // Default SIFT parameters
        int nOctaveLayers = 3;
        int contrastThreshold = 4;
        int edgeThreshold = 10;
        double sigma = 1.6;
};

void computeSIFTDescriptors(const cv::Mat& gray, 
        const std::vector<cv::KeyPoint>& keypoints, 
        cv::Mat& descriptors);