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
};

void onDoGTrackbar(int, void* userData);
void detectDoGCamera();