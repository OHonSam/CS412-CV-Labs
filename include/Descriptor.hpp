#pragma once
#include <vector>
#include <opencv2/opencv.hpp>

class IDescriptor {
public:
    virtual ~IDescriptor() {}
    virtual void compute(const cv::Mat& gray,
                         const std::vector<cv::KeyPoint>& kp,
                         cv::Mat& desc) = 0;
};
