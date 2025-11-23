#pragma once
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include "KeypointDetector.hpp"

struct BlobParams {
    // Thresholds
    int minThreshold = 10;
    int maxThreshold = 200;
    int thresholdStep = 10;

    // Filtering (Enable these to filter noise)
    bool filterByArea = true;
    int minArea = 100;
    int maxArea = 10000;
    float minDistBetweenBlobs = 10.0f;
    bool filterByCircularity = false;
    bool filterByConvexity = false;
    bool filterByInertia = false;

    // Maximum values for trackbars
    static constexpr int MAX_THRESHOLD = 255;
    static constexpr int MAX_THRESHOLD_STEP = 50;
    static constexpr int MAX_AREA = 10000;

    int getValidMinThreshold() const {
        return std::max(0, minThreshold);
    }

    int getValidMaxThreshold() const {
        return std::min(MAX_THRESHOLD, maxThreshold);
    }

    int getValidThresholdStep() const {
        return std::max(1, thresholdStep);
    }

    int getValidMinArea() const {
        return std::max(1, minArea);
    }

    int getValidMaxArea() const {
        return std::min(MAX_AREA, maxArea);
    }
};


struct BlobContext {
    cv::Mat src, gray;
    BlobParams params;
};

class BlobDetector : public IKeypointDetector {
    private:
        BlobContext context;
    public:
        std::vector<cv::KeyPoint> detect(const cv::Mat& gray) override;
        void createTrackbars(const std::string& win, void* userdata) override;
};

void detectBlob(const std::string& imagePath);
void detectBlobCamera();
void onBlobTrackbar(int, void* userData);
std::vector<cv::KeyPoint> myBlobDetection(const cv::Mat& gray, const BlobContext& context);