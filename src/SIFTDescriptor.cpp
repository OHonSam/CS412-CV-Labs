#include "SIFTDescriptor.hpp"
#include <opencv2/opencv.hpp>

void computeSIFTDescriptors(const cv::Mat& gray, 
                            const std::vector<cv::KeyPoint>& keypoints, 
                            cv::Mat& descriptors) {
    cv::Ptr<cv::SIFT> sift = cv::SIFT::create();
    sift->compute(gray, const_cast<std::vector<cv::KeyPoint>&>(keypoints), descriptors);
}