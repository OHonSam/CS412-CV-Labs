#include "SIFTDescriptor.hpp"
#include <opencv2/opencv.hpp>

void computeSIFTDescriptors(const cv::Mat& gray, 
                            const std::vector<cv::KeyPoint>& keypoints, 
                            cv::Mat& descriptors,
                            const SIFTContext& context) {

    // Pass the parameters from SIFTContext to the SIFT detector
    cv::Ptr<cv::SIFT> sift = cv::SIFT::create(
        context.nFeatures,
        context.nOctaveLayers,
        context.getContrastThreshold(),
        context.getEdgeThreshold(),
        context.getSigma()
    );

    sift->compute(gray, const_cast<std::vector<cv::KeyPoint>&>(keypoints), descriptors);
}

void createSIFTTrackbars(const std::string& windowName, SIFTContext* ctx, void (*onCallback)(int, void*)) {
    cv::createTrackbar("SIFT nFeatures", windowName, &ctx->nFeatures, 1000, onCallback, ctx);
    cv::createTrackbar("SIFT nOctaveLayers", windowName, &ctx->nOctaveLayers, 10, onCallback, ctx);
    cv::createTrackbar("SIFT ContrastThreshold x100", windowName, &ctx->contrastThreshold_x100, 100, onCallback, ctx);
    cv::createTrackbar("SIFT EdgeThreshold x10", windowName, &ctx->edgeThreshold_x10, 200, onCallback, ctx);
    cv::createTrackbar("SIFT Sigma x10", windowName, &ctx->sigma_x10, 50, onCallback, ctx);
}