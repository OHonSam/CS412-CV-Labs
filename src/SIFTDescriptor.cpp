#include "SIFTDescriptor.hpp"

void SIFTDescriptor::compute(const cv::Mat& gray,
                             const std::vector<cv::KeyPoint>& keypoints,
                             cv::Mat& descriptors) {
    sift->compute(gray, const_cast<std::vector<cv::KeyPoint>&>(keypoints), descriptors);
}