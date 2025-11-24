#pragma once
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>


struct DoGContext {
    cv::Mat src;
    cv::Mat gray;
    int sigma1 = 1;  // Standard deviation for the first Gaussian
    int sigmaDiff = 1;  // Difference in standard deviation for the second Gaussian
    int kernelSize = 5;  // Size of the Gaussian kernel
    int threshold = 150; // Default value (0-255)

    int getOddKernelSize() const {
        int odd_ksize = (kernelSize / 2) * 2 + 1;
        return std::max(3, odd_ksize);
    }

    int getSafeSigmaDiff() const {
        return std::max(1, sigmaDiff);
    }
};

void onDoGTrackbar(int, void* userData);
void createDoGTrackbars(const std::string& windowName, DoGContext* ctx, void (*onCallback)(int, void*));
std::vector<cv::KeyPoint> getDoGKeypoints(const cv::Mat& dogResponse, float threshold);
void detectDoG(const std::string& imagePath);
void detectDoGCamera();