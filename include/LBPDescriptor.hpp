#pragma once
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

void computeLBPDescriptors(const cv::Mat& gray, 
        const std::vector<cv::KeyPoint>& keypoints, 
        cv::Mat& descriptors);