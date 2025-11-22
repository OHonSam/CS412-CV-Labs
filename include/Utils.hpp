#pragma once
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>


// Non-Maximum Suppression for keypoints
std::vector<cv::KeyPoint> keypointNMS(const std::vector<cv::KeyPoint>& keypoints, double minDistance);

// UI and Helper functions
void displayHelp();

// Convolution and Filter functions
cv::Mat applyConvolution(const cv::Mat& src, const cv::Mat& kernel); 
std::vector<float> createGaussianFilter1D(int size); 
cv::Mat applyHorizontalConvolution1D(const cv::Mat& src, const std::vector<float>& horizontalKernel1D);
cv::Mat applyVerticalConvolution1D(const cv::Mat& src, const std::vector<float>& verticalKernel1D);
cv::Mat applySeparableConvolution(const cv::Mat& src, const std::vector<float>& kernel);