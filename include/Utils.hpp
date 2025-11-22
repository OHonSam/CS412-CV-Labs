#pragma once
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>


// UI and Helper functions
void displayHelp();
void openCamera();

// Convolution and Filter functions
cv::Mat applyConvolution(const cv::Mat& src, const cv::Mat& kernel); 
std::vector<float> createGaussianFilter1D(int size); 
cv::Mat applyHorizontalConvolution1D(const cv::Mat& src, const std::vector<float>& horizontalKernel1D);
cv::Mat applyVerticalConvolution1D(const cv::Mat& src, const std::vector<float>& verticalKernel1D);
cv::Mat applySeparableConvolution(const cv::Mat& src, const std::vector<float>& kernel);