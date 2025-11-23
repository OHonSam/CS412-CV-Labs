#pragma once
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>


struct BlobContext {
    cv::Mat src;
    cv::Mat gray;

    // Thresholds
    int minThreshold = 50;
    int maxThreshold = 255;
    int thresholdStep = 5;
    
    // Filtering (Enable these to filter noise)
    bool filterByArea = true;
    bool filterByCircularity = true;
    bool filterByConvexity = true;
    bool filterByInertia = true;

    int minArea = 50;         
    int maxArea = 50000;
    float minDistBetweenBlobs = 10.0f;
    int minCircularity = 40; // in percentage (0-100)
    int minConvexity = 40;   // in percentage (0-100)
    int minInertia = 40;     // in percentage (0-100)

    int getSafeThresholdStep() const {
        return std::max(1, thresholdStep);
    }

    float getMinCircularity() const {
        return minCircularity / 100.0f;
    }

    float getMinConvexity() const {
        return minConvexity / 100.0f;
    }

    float getMinInertia() const {
        return minInertia / 100.0f;
    }
};

void detectBlob(const std::string& imagePath);
void detectBlobCamera();
void onBlobTrackbar(int, void* userData);
std::vector<cv::KeyPoint> myBlobDetection(const cv::Mat& gray, const BlobContext& context);
void detectDoG(const std::string& imagePath);