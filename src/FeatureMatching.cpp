#include "FeatureMatching.hpp"
#include "HarrisDetector.hpp"
#include "DoGDetector.hpp"
#include "BlobDetector.hpp"
#include <iostream>

void matchFeatures(const std::string& detectorType, const std::string& descriptorType, 
                  const std::string& img1Path, const std::string& img2Path) {
    std::cout << "Matching " << img1Path << " and " << img2Path << std::endl;
    std::cout << "Detector: " << detectorType << ", Descriptor: " << descriptorType << std::endl;
    if (detectorType != "harris" && detectorType != "dog" && detectorType != "blob") {
        std::cerr << "Error: Unknown detector type '" << detectorType << "'" << std::endl;
        return;
    }

    if (descriptorType != "sift" && descriptorType != "lbp") {
        std::cerr << "Error: Unknown descriptor type '" << descriptorType << "'" << std::endl;
        return;
    }

    // Load images
    cv::Mat img1 = cv::imread(img1Path, cv::IMREAD_COLOR);
    cv::Mat img2 = cv::imread(img2Path, cv::IMREAD_COLOR);
    if (img1.empty() || img2.empty()) {
        std::cerr << "Error: Could not read one of the images." << std::endl;
        return;
    }

    cv::Mat gray1, gray2;
    cv::cvtColor(img1, gray1, cv::COLOR_BGR2GRAY);
    cv::cvtColor(img2, gray2, cv::COLOR_BGR2GRAY);

    // Detect keypoints
    std::vector<cv::KeyPoint> keypoints1, keypoints2;
    cv::Mat dst1, dst2;
    if (detectorType == "harris") {
        HarrisContext ctx1, ctx2;
        ctx1.src = img1;
        ctx2.src = img2;

        ctx1.gray = gray1;
        ctx2.gray = gray2;

        cv::cornerHarris(ctx1.gray, dst1, ctx1.blockSize, ctx1.apertureSize, ctx1.k_x100 / 100.0);
        cv::cornerHarris(ctx2.gray, dst2, ctx2.blockSize, ctx2.apertureSize, ctx2.k_x100 / 100.0);

        cv::normalize(dst1, dst1);
        cv::normalize(dst2, dst2);

        keypoints1 = getHarrisKeypoints(dst1, ctx1.threshold);
        keypoints2 = getHarrisKeypoints(dst2, ctx2.threshold);

    } else if (detectorType == "dog") {
        DoGContext ctx1, ctx2;
        ctx1.src = img1;
        ctx2.src = img2;

        ctx1.gray = gray1;
        ctx2.gray = gray2;

        cv::Mat blur1_1, blur1_2, blur2_1, blur2_2;
        cv::GaussianBlur(ctx1.gray, blur1_1, cv::Size(ctx1.kernelSize, ctx1.kernelSize), ctx1.sigma1);
        cv::GaussianBlur(ctx2.gray, blur2_1, cv::Size(ctx2.kernelSize, ctx2.kernelSize), ctx2.sigma1);
        cv::GaussianBlur(ctx1.gray, blur1_2, cv::Size(ctx1.kernelSize, ctx1.kernelSize), ctx1.sigma1 + ctx1.sigmaDiff);
        cv::GaussianBlur(ctx2.gray, blur2_2, cv::Size(ctx2.kernelSize, ctx2.kernelSize), ctx2.sigma1 + ctx2.sigmaDiff);

        cv::subtract(blur1_1, blur1_2, dst1);
        cv::subtract(blur2_1, blur2_2, dst2);
        cv::normalize(dst1, dst1);
        cv::normalize(dst2, dst2);

        keypoints1 = getDoGKeypoints(dst1, ctx1.threshold);
        keypoints2 = getDoGKeypoints(dst2, ctx2.threshold);

    } else if (detectorType == "blob") {
        BlobContext ctx1, ctx2;
        ctx1.src = img1;
        ctx2.src = img2;

        ctx1.gray = gray1;
        ctx2.gray = gray2;

        cv::SimpleBlobDetector::Params params1, params2;
        params1.minThreshold = ctx1.minThreshold;
        params1.maxThreshold = ctx1.maxThreshold;
        params1.thresholdStep = ctx1.thresholdStep;
        params1.filterByArea = ctx1.filterByArea;
        params1.minArea = ctx1.minArea;
        params1.maxArea = ctx1.maxArea;
        params1.minDistBetweenBlobs = ctx1.minDistBetweenBlobs;

        params2.minThreshold = ctx2.minThreshold;
        params2.maxThreshold = ctx2.maxThreshold;
        params2.thresholdStep = ctx2.thresholdStep;
        params2.filterByArea = ctx2.filterByArea;
        params2.minArea = ctx2.minArea;
        params2.maxArea = ctx2.maxArea;
        params2.minDistBetweenBlobs = ctx2.minDistBetweenBlobs;

        cv::Ptr<cv::SimpleBlobDetector> detector1 = cv::SimpleBlobDetector::create(params1);
        cv::Ptr<cv::SimpleBlobDetector> detector2 = cv::SimpleBlobDetector::create(params2);
        detector1->detect(ctx1.gray, keypoints1);
        detector2->detect(ctx2.gray, keypoints2);
    }
}