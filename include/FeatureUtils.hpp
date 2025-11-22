#pragma once
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

struct HarrisContext {
    cv::Mat src;
    cv::Mat gray;
    int blockSize = 2;  // Size of neighborhood considered for corner detection
    int apertureSize = 3;  // Aperture parameter for the Sobel operator
    int k_x100 = 4;  // Harris detector free parameter
    int threshold = 200;  // Threshold for detecting corners
    int max_harris_blockSize = 10;
    int max_harris_ksize = 7;
    int max_harris_k_x100 = 10;
    int max_harris_threshold = 255;
};

struct BlobContext {
    cv::Mat src;
    cv::Mat gray;
    // Thresholds
    int minThreshold = 10;
    int maxThreshold = 200;
    int thresholdStep = 10;
    
    // Filtering (Enable these to filter noise)
    bool filterByArea = true;
    int minArea = 100;         // Trackbar variable
    int maxArea = 10000;
    
    bool filterByCircularity = false;
    bool filterByConvexity = false;
    bool filterByInertia = false;
};

// UI and Helper functions
void displayHelp();
void openCamera();

// Convolution and Filter functions
cv::Mat applyConvolution(const cv::Mat& src, const cv::Mat& kernel); 
std::vector<float> createGaussianFilter1D(int size); 
cv::Mat applyHorizontalConvolution1D(const cv::Mat& src, const std::vector<float>& horizontalKernel1D);
cv::Mat applyVerticalConvolution1D(const cv::Mat& src, const std::vector<float>& verticalKernel1D);
cv::Mat applySeparableConvolution(const cv::Mat& src, const std::vector<float>& kernel);

// Feature Detection functions
void detectHarris(const std::string& imagePath);
void detectHarrisCamera();
void onHarrisTrackbar(int, void* userData);
void detectBlob(const std::string& imagePath);
void detectBlobCamera();
void detectDoG(const std::string& imagePath);
void detectDoGCamera();

// Matching functions
void matchFeatures(const std::string& detectorType, const std::string& descriptorType, 
                  const std::string& img1Path, const std::string& img2Path);