#pragma once
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include "KeypointDetector.hpp"

struct DoGParams {
    int sigma1 = 1;        // Standard deviation for the first Gaussian
    int sigmaDiff = 1;     // Difference in standard deviation for the second Gaussian
    int kernelSize = 5;    // Size of the Gaussian kernel
    float threshold = 200; // Response threshold for keypoint detection

    // Maximum values for trackbars
    static constexpr int MAX_SIGMA = 100;
    static constexpr int MAX_KERNEL_SIZE = 21;
    static constexpr int MAX_SIGMA_DIFF = 100;

    int getValidKernelSize() const {
        int odd = (kernelSize / 2) * 2 + 1;
        return std::max(3, odd);
    }

    int getValidSigmaDiff() const {
        return std::max(1, sigmaDiff);
    }
};

struct DoGContext {
    cv::Mat src, gray;
    DoGParams params;
};

class DoGDetector : public IKeypointDetector {
    private:
        DoGContext context;
    public:
        std::vector<cv::KeyPoint> detect(const cv::Mat& gray) override;
        void createTrackbars(const std::string& win, void* userdata) override;
};

void onDoGTrackbar(int, void* userData);
std::vector<cv::KeyPoint> getDoGKeypoints(const cv::Mat& dogResponse, float threshold);
void detectDoG(const std::string& imagePath);
void detectDoGCamera();