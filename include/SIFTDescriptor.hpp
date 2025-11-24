#pragma once
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

struct SIFTContext {
        // Default SIFT parameters
        int nFeatures = 0;
        int nOctaveLayers = 3;
        int contrastThreshold_x100 = 4;
        int edgeThreshold_x10 = 100;
        int sigma_x10 = 16;

        double getSigma() const {
            return sigma_x10 / 10.0;
        }

        double getContrastThreshold() const {
            return contrastThreshold_x100 / 100.0;
        }

        double getEdgeThreshold() const {
            return edgeThreshold_x10 / 10.0;
        }
};

void createSIFTTrackbars(const std::string& windowName, SIFTContext* ctx, void (*onCallback)(int, void*));

void computeSIFTDescriptors(const cv::Mat& gray, 
        const std::vector<cv::KeyPoint>& keypoints, 
        cv::Mat& descriptors, const SIFTContext& context);