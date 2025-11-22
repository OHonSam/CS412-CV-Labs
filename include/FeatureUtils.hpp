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

// UI and Helper functions
void displayHelp();
void openCamera();

// Feature Detection functions
void detectHarris(const std::string& imagePath);
void onHarrisTrackbar(int, void* userData);
void detectBlob(const std::string& imagePath);
void detectDoG(const std::string& imagePath);

// Matching functions
void matchFeatures(const std::string& detectorType, const std::string& descriptorType, 
                  const std::string& img1Path, const std::string& img2Path);