#pragma once
#include <vector>
#include <opencv2/opencv.hpp>

class IKeypointDetector {
public:
    virtual ~IKeypointDetector() {}
    virtual std::vector<cv::KeyPoint> detect(const cv::Mat& gray) = 0;
    virtual void createTrackbars(const std::string& win, void* userdata) {}
};
