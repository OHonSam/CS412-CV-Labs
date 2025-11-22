#pragma once
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>


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
    float minDistBetweenBlobs = 10.0f;
    
    bool filterByCircularity = false;
    bool filterByConvexity = false;
    bool filterByInertia = false;
};

void detectBlob(const std::string& imagePath);
void detectBlobCamera();
void onBlobTrackbar(int, void* userData);
std::vector<cv::KeyPoint> myBlobDetection(const cv::Mat& gray, const BlobContext& context);
void detectDoG(const std::string& imagePath);