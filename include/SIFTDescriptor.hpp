#pragma once
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include "Descriptor.hpp"

class SIFTDescriptor : public IDescriptor {
    private:
        cv::Ptr<cv::SIFT> sift;
        
    public:
        SIFTDescriptor() {
            sift = cv::SIFT::create();
        }

        void compute(const cv::Mat& gray,
                    const std::vector<cv::KeyPoint>& keypoints,
                    cv::Mat& descriptors) override;
};