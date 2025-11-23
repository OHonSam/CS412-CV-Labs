#include "FeatureMatching.hpp"
#include "HarrisDetector.hpp"
#include "DoGDetector.hpp"
#include "BlobDetector.hpp"
#include <iostream>

void computeSIFTDescriptors(const cv::Mat& gray, 
                            const std::vector<cv::KeyPoint>& keypoints, 
                            cv::Mat& descriptors) {
    cv::Ptr<cv::SIFT> sift = cv::SIFT::create();
    sift->compute(gray, const_cast<std::vector<cv::KeyPoint>&>(keypoints), descriptors);
}

void computeLBPDescriptors(const cv::Mat& gray, 
                            const std::vector<cv::KeyPoint>& keypoints, 
                            cv::Mat& descriptors) {
    const int radius = 1;
    const int patchSize = 16; // Size of the region around keypoint
    descriptors = cv::Mat::zeros(static_cast<int>(keypoints.size()), 256, CV_32F);

    for (size_t i = 0; i < keypoints.size(); ++i) {
        const cv::KeyPoint& kp = keypoints[i];
        int cx = static_cast<int>(kp.pt.x);
        int cy = static_cast<int>(kp.pt.y);
        
        // Define patch boundaries (with safety checks)
        int x1 = std::max(radius, cx - patchSize / 2);
        int y1 = std::max(radius, cy - patchSize / 2);
        int x2 = std::min(gray.cols - radius - 1, cx + patchSize / 2);
        int y2 = std::min(gray.rows - radius - 1, cy + patchSize / 2);

        // Compute LBP histogram for the patch
        for (int y = y1; y < y2; ++y) {
            for (int x = x1; x < x2; ++x) {
                uchar center = gray.at<uchar>(y, x);
                uchar lbpCode = 0;

                // 8 neighbors (same as computeLBPImage)
                lbpCode |= (gray.at<uchar>(y - 1, x - 1) >= center) << 7;
                lbpCode |= (gray.at<uchar>(y - 1, x) >= center) << 6;
                lbpCode |= (gray.at<uchar>(y - 1, x + 1) >= center) << 5;
                lbpCode |= (gray.at<uchar>(y, x + 1) >= center) << 4;
                lbpCode |= (gray.at<uchar>(y + 1, x + 1) >= center) << 3;
                lbpCode |= (gray.at<uchar>(y + 1, x) >= center) << 2;
                lbpCode |= (gray.at<uchar>(y + 1, x - 1) >= center) << 1;
                lbpCode |= (gray.at<uchar>(y, x - 1) >= center) << 0;

                descriptors.at<float>(static_cast<int>(i), lbpCode) += 1.0f;
            }
        }

        cv::normalize(descriptors.row(i), descriptors.row(i), 1.0, 0.0, cv::NORM_L1);
    }
}

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

        cv::Mat dst1_norm, dst2_norm;
        cv::normalize(dst1, dst1_norm, 0, 255, cv::NORM_MINMAX, CV_32FC1);
        cv::normalize(dst2, dst2_norm, 0, 255, cv::NORM_MINMAX, CV_32FC1);

        keypoints1 = getHarrisKeypoints(dst1_norm, ctx1.threshold);
        keypoints2 = getHarrisKeypoints(dst2_norm, ctx2.threshold);

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

        cv::Mat dst1_norm, dst2_norm;
        cv::normalize(dst1, dst1_norm, 0, 255, cv::NORM_MINMAX, CV_32FC1);
        cv::normalize(dst2, dst2_norm, 0, 255, cv::NORM_MINMAX, CV_32FC1);

        keypoints1 = getDoGKeypoints(dst1_norm, ctx1.threshold);
        keypoints2 = getDoGKeypoints(dst2_norm, ctx2.threshold);

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

    // Compute descriptors
    cv::Mat descriptors1, descriptors2;
    if (descriptorType == "sift") {
        computeSIFTDescriptors(gray1, keypoints1, descriptors1);
        computeSIFTDescriptors(gray2, keypoints2, descriptors2);
    } else if (descriptorType == "lbp") {
        computeLBPDescriptors(gray1, keypoints1, descriptors1);
        computeLBPDescriptors(gray2, keypoints2, descriptors2);
    }

    if (descriptors1.empty() || descriptors2.empty()) {
        std::cerr << "Error: No descriptors computed." << std::endl;
        return;
    }

    std::cout << "Keypoints: " << keypoints1.size() << " (img1), " 
              << keypoints2.size() << " (img2)" << std::endl;

    // Match descriptors using KNN matcher
    cv::BFMatcher matcher(cv::NORM_L2); // Remove cross-check for KNN
    std::vector<std::vector<cv::DMatch>> knn_matches;
    matcher.knnMatch(descriptors1, descriptors2, knn_matches, 2); // Find 2 nearest neighbors

    // Apply Lowe's ratio test to filter good matches
    const float ratio_thresh = 0.75f; // Standard threshold

    // Visualize all matches from KNN
    std::vector<cv::DMatch> all_matches;
    for (const auto& knn_match : knn_matches) {
        if (!knn_match.empty()) {
            all_matches.push_back(knn_match[0]);
        }
    }

    // // Visualize good matches
    // std::vector<cv::DMatch> good_matches;
    // for (size_t i = 0; i < knn_matches.size(); i++) {
    //     // Only consider if we found 2 neighbors
    //     if (knn_matches[i].size() == 2) {
    //         // If the best match is significantly better than the second-best
    //         if (knn_matches[i][0].distance < ratio_thresh * knn_matches[i][1].distance) {
    //             good_matches.push_back(knn_matches[i][0]);
    //         }
    //     }
    // }

    // std::cout << "Good matches after Lowe's ratio test: " << good_matches.size() << std::endl;

    // if (good_matches.empty()) {
    //     std::cerr << "No good matches found." << std::endl;
    //     return;
    // }

    cv::Mat imgMatches;
    cv::drawMatches(img1, keypoints1, img2, keypoints2, all_matches, imgMatches,
                    cv::Scalar::all(-1), cv::Scalar::all(-1),
                    std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

    // Resize the visualization to fit the screen
    cv::Mat imgMatchesResized;
    double scale = 0.5;
    cv::resize(imgMatches, imgMatchesResized, cv::Size(), scale, scale);

    cv::namedWindow("Feature Matches", cv::WINDOW_AUTOSIZE);
    cv::imshow("Feature Matches", imgMatchesResized);
    cv::waitKey(0);

    cv::destroyAllWindows();
}