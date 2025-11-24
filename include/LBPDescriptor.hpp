#pragma once
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

struct LBPContext {
        int radius = 1;  // Radius for LBP
        int patchSize = 16; // Size of the patch around keypoint
};

void createLBPTrackbars(const std::string& windowName, LBPContext* ctx, void (*onCallback)(int, void*));
void computeLBPDescriptors(const cv::Mat& gray, 
        const std::vector<cv::KeyPoint>& keypoints, 
        cv::Mat& descriptors, const LBPContext& context);