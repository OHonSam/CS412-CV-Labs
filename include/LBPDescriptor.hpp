#pragma once
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include "Descriptor.hpp"

class LBPDescriptor : public IDescriptor {
public:
    void compute(const cv::Mat& gray,
                 const std::vector<cv::KeyPoint>& keypoints,
                 cv::Mat& descriptors) override;
};