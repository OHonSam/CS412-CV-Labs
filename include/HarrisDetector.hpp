#pragma once
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include "KeypointDetector.hpp"

/**
 * Configuration parameters for Harris Corner Detection
 */
struct HarrisParams {
    int blockSize = 2;          // Neighborhood size for corner detection
    int apertureSize = 3;       // Sobel operator aperture (must be odd)
    int k_x100 = 4;             // Harris free parameter k * 100 (k = 0.04)
    int threshold = 200;        // Corner detection threshold
    
    // Maximum values for trackbars
    static constexpr int MAX_BLOCK_SIZE = 10;
    static constexpr int MAX_APERTURE_SIZE = 7;
    static constexpr int MAX_K_VALUE = 10;
    static constexpr int MAX_THRESHOLD = 255;
    
    // Validation
    int getValidBlockSize() const { return std::max(2, blockSize); }
    int getValidApertureSize() const { 
        int odd = (apertureSize / 2) * 2 + 1;
        return std::max(3, odd);
    }
    double getK() const { return k_x100 / 100.0; }
};

/**
 * Context for Harris detection with UI state
 */
struct HarrisContext {
    cv::Mat src;
    cv::Mat gray;
    HarrisParams params;
};

/**
 * Harris Corner Detector implementation
 */
class HarrisDetector : public IKeypointDetector {
private:
    HarrisContext context;
    
    cv::Mat computeHarrisResponse(const cv::Mat& gray, const HarrisParams& params);
    std::vector<cv::KeyPoint> extractKeypoints(const cv::Mat& response, float threshold);

public:
    std::vector<cv::KeyPoint> detect(const cv::Mat& gray) override;
    void createTrackbars(const std::string& win, void* userdata) override;
};

// Public API functions
void detectHarris(const std::string& imagePath);
void detectHarrisCamera();

// Helper functions
void onHarrisTrackbar(int, void* userData);
void onHarrisMatchingTrackbar(int, void* userData);
void myCornerHarris(const cv::Mat& src, cv::Mat& dst, int blockSize, 
                    int apertureSize, double k);
std::vector<cv::KeyPoint> getHarrisKeypoints(const cv::Mat& harrisResponse, 
                                             float threshold);