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

// void onMatchTrackbar(int, void* userData) {
//     MatchContext* ctx = static_cast<MatchContext*>(userData);

//     std::vector<cv::KeyPoint> keypoints1, keypoints2;

//     if (ctx->detectorType == "harris") {
//         // Reuse Harris logic for Image 1
//         int safe_block = std::max(2, ctx->harris_blockSize);
//         int odd_aperture = (ctx->harris_apertureSize / 2) * 2 + 1;
        
//         cv::Mat dst1, dst1_norm;
//         cv::cornerHarris(ctx->gray1, dst1, safe_block, odd_aperture, ctx->harris_k_x100 / 100.0);
//         cv::normalize(dst1, dst1_norm, 0, 255, cv::NORM_MINMAX, CV_32FC1);
//         keypoints1 = getHarrisKeypoints(dst1_norm, (float)ctx->harris_threshold);

//         // Reuse Harris logic for Image 2
//         cv::Mat dst2, dst2_norm;
//         cv::cornerHarris(ctx->gray2, dst2, safe_block, odd_aperture, ctx->harris_k_x100 / 100.0);
//         cv::normalize(dst2, dst2_norm, 0, 255, cv::NORM_MINMAX, CV_32FC1);
//         keypoints2 = getHarrisKeypoints(dst2_norm, (float)ctx->harris_threshold);
//     }
//     else if (ctx->detectorType == "dog") {
//         // Ensure kernel size is odd and at least 3
//         int ksize = ctx->dog_kernelSize;
//         ksize = (ksize / 2) * 2 + 1;
//         if (ksize < 3) ksize = 3;

//         // Ensure sigma difference is at least 1
//         int sigmaDiff = ctx->dog_sigmaDiff;
//         if (sigmaDiff < 1) sigmaDiff = 1;
//         // Reuse DoG logic for Image 1
//         cv::Mat blur1_1, blur1_2, dog1, dog1_norm;
//         cv::GaussianBlur(ctx->gray1, blur1_1, cv::Size(ksize, ksize), ctx->dog_sigma1);
//         cv::GaussianBlur(ctx->gray1, blur1_2, cv::Size(ksize, ksize), ctx->dog_sigma1 + sigmaDiff);
//         cv::subtract(blur1_1, blur1_2, dog1);
//         cv::normalize(dog1, dog1_norm, 0, 255, cv::NORM_MINMAX, CV_32FC1);
//         keypoints1 = getDoGKeypoints(dog1_norm, (float)ctx->dog_threshold);

//         // Reuse DoG logic for Image 2
//         cv::Mat blur2_1, blur2_2, dog2, dog2_norm;
//         cv::GaussianBlur(ctx->gray2, blur2_1, cv::Size(ksize, ksize), ctx->dog_sigma1);
//         cv::GaussianBlur(ctx->gray2, blur2_2, cv::Size(ksize, ksize), ctx->dog_sigma1 + sigmaDiff);
//         cv::subtract(blur2_1, blur2_2, dog2);
//         cv::normalize(dog2, dog2_norm, 0, 255, cv::NORM_MINMAX, CV_32FC1);
//         keypoints2 = getDoGKeypoints(dog2_norm, (float)ctx->dog_threshold);
//     }
//     else if (ctx->detectorType == "blob") {
//         cv::SimpleBlobDetector::Params blobParams1, blobParams2;

//         blobParams1.filterByArea = ctx->blob_minArea;
//         blobParams1.filterByCircularity = ctx->blob_maxThreshold;
//         blobParams1.filterByConvexity = ctx->blob_minThreshold;
//         blobParams1.filterByInertia = ctx->blob_maxThreshold;
//         blobParams1.minThreshold = ctx->blob_minThreshold;
//         blobParams1.maxThreshold = ctx->blob_maxThreshold;
//         blobParams1.thresholdStep = ctx->blob_thresholdStep;

//         cv::Ptr<cv::SimpleBlobDetector> detector1 = cv::SimpleBlobDetector::create(blobParams1);
//         detector1->detect(ctx->gray1, keypoints1);

//         blobParams2.filterByArea = ctx->blob_minArea;
//         blobParams2.filterByCircularity = ctx->blob_maxThreshold;
//         blobParams2.filterByConvexity = ctx->blob_minThreshold;
//         blobParams2.filterByInertia = ctx->blob_maxThreshold;
//         blobParams2.minThreshold = ctx->blob_minThreshold;
//         blobParams2.maxThreshold = ctx->blob_maxThreshold;
//         blobParams2.thresholdStep = ctx->blob_thresholdStep;

//         cv::Ptr<cv::SimpleBlobDetector> detector2 = cv::SimpleBlobDetector::create(blobParams2);
//         detector2->detect(ctx->gray2, keypoints2);
//     }

//     if (keypoints1.empty() || keypoints2.empty()) return;

//     // ---------------------------------------------------------
//     // 2. COMPUTE DESCRIPTORS
//     // ---------------------------------------------------------
//     cv::Mat descriptors1, descriptors2;
//     if (ctx->descriptorType == "sift") {
//         computeSIFTDescriptors(ctx->gray1, keypoints1, descriptors1);
//         computeSIFTDescriptors(ctx->gray2, keypoints2, descriptors2);
//     } else {
//         computeLBPDescriptors(ctx->gray1, keypoints1, descriptors1);
//         computeLBPDescriptors(ctx->gray2, keypoints2, descriptors2);
//     }

//     // ---------------------------------------------------------
//     // 3. MATCH & DRAW
//     // ---------------------------------------------------------
//     cv::BFMatcher matcher(cv::NORM_L2);
//     std::vector<std::vector<cv::DMatch>> knn_matches;
//     matcher.knnMatch(descriptors1, descriptors2, knn_matches, 2);

//     std::vector<cv::DMatch> good_matches;
//     float ratio = ctx->ratioTest_x100 / 100.0f;
//     for (auto& m : knn_matches) {
//         if (m.size() == 2 && m[0].distance < ratio * m[1].distance) {
//             good_matches.push_back(m[0]);
//         }
//     }

//     cv::Mat imgMatches;
//     cv::drawMatches(ctx->img1, keypoints1, ctx->img2, keypoints2, good_matches, imgMatches, 
//                     cv::Scalar::all(-1), cv::Scalar::all(-1), std::vector<char>(), 
//                     cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
        
//     // Resize for display
//     cv::Mat resized;
//     cv::resize(imgMatches, resized, cv::Size(), 0.5, 0.5);
//     cv::imshow("Feature Matches", resized);
// }

// void matchFeatures(const std::string& detectorType, const std::string& descriptorType, 
//                   const std::string& img1Path, const std::string& img2Path) {
//     std::cout << "Matching " << img1Path << " and " << img2Path << std::endl;
//     std::cout << "Detector: " << detectorType << ", Descriptor: " << descriptorType << std::endl;
//     if (detectorType != "harris" && detectorType != "dog" && detectorType != "blob") {
//         std::cerr << "Error: Unknown detector type '" << detectorType << "'" << std::endl;
//         return;
//     }

//     if (descriptorType != "sift" && descriptorType != "lbp") {
//         std::cerr << "Error: Unknown descriptor type '" << descriptorType << "'" << std::endl;
//         return;
//     }

//     const std::string winName = "Feature Matches";
//     cv::namedWindow(winName, cv::WINDOW_NORMAL);

//     MatchContext* ctx = new MatchContext();

//     // Load detector and descriptor types
//     ctx->detectorType = detectorType;
//     ctx->descriptorType = descriptorType;

//     // Load images
//     ctx->img1 = cv::imread(img1Path, cv::IMREAD_COLOR);
//     ctx->img2 = cv::imread(img2Path, cv::IMREAD_COLOR);
//     if (ctx->img1.empty() || ctx->img2.empty()) {
//         std::cerr << "Error: Could not read one of the images." << std::endl;
//         return;
//     }

//     cv::cvtColor(ctx->img1, ctx->gray1, cv::COLOR_BGR2GRAY);
//     cv::cvtColor(ctx->img2, ctx->gray2, cv::COLOR_BGR2GRAY);

//     // Detect keypoints
//     std::vector<cv::KeyPoint> keypoints1, keypoints2;
//     cv::Mat dst1, dst2;
//     if (detectorType == "harris") {
//         cv::createTrackbar("Block Size", winName, &ctx->harris_blockSize, 10, onMatchTrackbar, ctx);
//         cv::createTrackbar("Aperture", winName, &ctx->harris_apertureSize, 7, onMatchTrackbar, ctx);
//         cv::createTrackbar("k x100", winName, &ctx->harris_k_x100, 100, onMatchTrackbar, ctx);
//         cv::createTrackbar("Threshold", winName, &ctx->harris_threshold, 255, onMatchTrackbar, ctx);

//     } else if (detectorType == "dog") {
//         cv::createTrackbar("Kernel Size", winName, &ctx->dog_kernelSize, 31, onMatchTrackbar, ctx);
//         cv::createTrackbar("Sigma1", winName, &ctx->dog_sigma1, 10, onMatchTrackbar, ctx);
//         cv::createTrackbar("Sigma Diff", winName, &ctx->dog_sigmaDiff, 10, onMatchTrackbar, ctx);
//         cv::createTrackbar("Threshold", winName, &ctx->dog_threshold, 255, onMatchTrackbar, ctx);

//     } else if (detectorType == "blob") {
//         cv::createTrackbar("Min Threshold", winName, &ctx->blob_minThreshold, 255, onMatchTrackbar, ctx);
//         cv::createTrackbar("Max Threshold", winName, &ctx->blob_maxThreshold, 255, onMatchTrackbar, ctx);
//         cv::createTrackbar("Threshold Step", winName, &ctx->blob_thresholdStep, 50, onMatchTrackbar, ctx);
//         cv::createTrackbar("Min Area", winName, &ctx->blob_minArea, 10000, onMatchTrackbar, ctx);
//     }

//     cv::createTrackbar("Ratio Test (x100)", winName, &ctx->ratioTest_x100, 100, onMatchTrackbar, ctx);

//     onMatchTrackbar(0, ctx);

//     cv::waitKey(0);
//     cv::destroyAllWindows();
//     delete ctx;
// }

void onMatchTrackbar(int, void* userdata) {
    auto* ctx = static_cast<MatchContext*>(userdata);

    if (ctx->gray1.empty() || ctx->gray2.empty()) return;

    auto kp1 = ctx->detector->detect(ctx->gray1);
    auto kp2 = ctx->detector->detect(ctx->gray2);

    if (kp1.empty() || kp2.empty()) return;

    cv::Mat d1, d2;
    ctx->descriptor->compute(ctx->gray1, kp1, d1);
    ctx->descriptor->compute(ctx->gray2, kp2, d2);

    cv::BFMatcher matcher(cv::NORM_L2);
    std::vector<std::vector<cv::DMatch>> knn;
    matcher.knnMatch(d1, d2, knn, 2);

    float ratio = ctx->ratioTest_x100 / 100.0f;
    std::vector<cv::DMatch> good;
    for (auto& m : knn)
        if (m.size() == 2 && m[0].distance < ratio * m[1].distance)
            good.push_back(m[0]);

    cv::Mat out;
    cv::drawMatches(ctx->img1, kp1, ctx->img2, kp2, good, out);
    cv::imshow("Feature Matches", out);
}

void matchFeatures(const std::string& detectorType, const std::string& descriptorType, 
                  const std::string& img1Path, const std::string& img2Path) {
    std::cout << "Matching " << img1Path << " and " << img2Path << std::endl;
    std::cout << "Detector: " << detectorType << ", Descriptor: " << descriptorType << std::endl;

    MatchContext* ctx = new MatchContext();

    // Load images
    ctx->img1 = cv::imread(img1Path, cv::IMREAD_COLOR);
    ctx->img2 = cv::imread(img2Path, cv::IMREAD_COLOR);
    if (ctx->img1.empty() || ctx->img2.empty()) {
        std::cerr << "Error: Could not read one of the images." << std::endl;
        delete ctx;
        return;
    }

    cv::cvtColor(ctx->img1, ctx->gray1, cv::COLOR_BGR2GRAY);
    cv::cvtColor(ctx->img2, ctx->gray2, cv::COLOR_BGR2GRAY);

    // Create detector
    if (detectorType == "harris") {
        ctx->detector = new HarrisDetector();
    } else if (detectorType == "dog") {
        ctx->detector = new DoGDetector();
    } else if (detectorType == "blob") {
        ctx->detector = new BlobDetector();
    } else {
        std::cerr << "Error: Unknown detector type '" << detectorType << "'" << std::endl;
        delete ctx;
        return;
    }

    // Create descriptor
    if (descriptorType == "sift") {
        ctx->descriptor = new SIFTDescriptor();
    } else if (descriptorType == "lbp") {
        ctx->descriptor = new LBPDescriptor();
    } else {
        std::cerr << "Error: Unknown descriptor type '" << descriptorType << "'" << std::endl;
        delete ctx->detector;
        delete ctx;
        return;
    }

    const std::string winName = "Feature Matches";
    cv::namedWindow(winName, cv::WINDOW_NORMAL);

    ctx->detector->createTrackbars(winName, ctx);
    cv::createTrackbar("Ratio Test (x100)", winName, &ctx->ratioTest_x100, 100, onMatchTrackbar, ctx);

    onMatchTrackbar(0, ctx);

    cv::waitKey(0);
    cv::destroyAllWindows();
    delete ctx;
}