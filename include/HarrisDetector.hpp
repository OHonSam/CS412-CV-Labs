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

    int getOddAperture() const {
        int odd_aperture = (apertureSize / 2) * 2 + 1;
        return std::max(3, odd_aperture);
    }

    int getSafeBlock() const {
        return std::max(2, blockSize);
    }

    double getK() const {
        return k_x100 / 100.0;
    }
};

void detectHarris(const std::string& imagePath);
void detectHarrisCamera();
std::vector<cv::KeyPoint> getHarrisKeypoints(const cv::Mat& harrisResponse, float threshold);
void onHarrisTrackbar(int, void* userData);
void myCornerHarris(const cv::Mat& src, cv::Mat& dst, int blockSize, int apertureSize, double k);