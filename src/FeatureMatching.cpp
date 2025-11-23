#include "FeatureMatching.hpp"
#include "HarrisDetector.hpp"
#include "DoGDetector.hpp"
#include "BlobDetector.hpp"
#include "SIFTDescriptor.hpp"
#include "LBPDescriptor.hpp"
#include <iostream>

void onMatchTrackbar(int, void* userData) {
    MatchContext* context = static_cast<MatchContext*>(userData);
    if (!context) return;   
    
    // Detect keypoints
    std::vector<cv::KeyPoint> keypoints1, keypoints2;
    cv::Mat dst1, dst2;
    if (context->detectorType == "harris") {
        cv::cornerHarris(
            context->gray1, dst1, 
            context->harrisContext.getSafeBlock(), context->harrisContext.getOddAperture(), context->harrisContext.getK()
        );
        cv::cornerHarris(
            context->gray2, dst2, 
            context->harrisContext.getSafeBlock(), context->harrisContext.getOddAperture(), context->harrisContext.getK()
        );

        cv::Mat dst1_norm, dst2_norm;
        cv::normalize(dst1, dst1_norm, 0, 255, cv::NORM_MINMAX, CV_32FC1);
        cv::normalize(dst2, dst2_norm, 0, 255, cv::NORM_MINMAX, CV_32FC1);

        keypoints1 = getHarrisKeypoints(dst1_norm, context->harrisContext.threshold);
        keypoints2 = getHarrisKeypoints(dst2_norm, context->harrisContext.threshold);

    } else if (context->detectorType == "dog") {
        cv::Mat blur1_1, blur1_2, blur2_1, blur2_2;
        cv::GaussianBlur(context->gray1, blur1_1, cv::Size(context->dogContext.getOddKernelSize(), context->dogContext.getOddKernelSize()), context->dogContext.sigma1);
        cv::GaussianBlur(context->gray2, blur2_1, cv::Size(context->dogContext.getOddKernelSize(), context->dogContext.getOddKernelSize()), context->dogContext.sigma1);
        cv::GaussianBlur(context->gray1, blur1_2, cv::Size(context->dogContext.getOddKernelSize(), context->dogContext.getOddKernelSize()), context->dogContext.sigma1 + context->dogContext.getSafeSigmaDiff());
        cv::GaussianBlur(context->gray2, blur2_2, cv::Size(context->dogContext.getOddKernelSize(), context->dogContext.getOddKernelSize()), context->dogContext.sigma1 + context->dogContext.getSafeSigmaDiff());

        cv::subtract(blur1_1, blur1_2, dst1);
        cv::subtract(blur2_1, blur2_2, dst2);

        cv::Mat dst1_norm, dst2_norm;
        cv::normalize(dst1, dst1_norm, 0, 255, cv::NORM_MINMAX, CV_32FC1);
        cv::normalize(dst2, dst2_norm, 0, 255, cv::NORM_MINMAX, CV_32FC1);

        keypoints1 = getDoGKeypoints(dst1_norm, context->dogContext.threshold);
        keypoints2 = getDoGKeypoints(dst2_norm, context->dogContext.threshold);

    } else if (context->detectorType == "blob") {
        cv::SimpleBlobDetector::Params params1, params2;
        params1.minThreshold = context->blobContext.minThreshold;
        params1.maxThreshold = context->blobContext.maxThreshold;
        params1.thresholdStep = context->blobContext.thresholdStep;
        params1.filterByArea = context->blobContext.filterByArea;
        params1.minArea = context->blobContext.minArea;
        params1.maxArea = context->blobContext.maxArea;
        params1.minDistBetweenBlobs = context->blobContext.minDistBetweenBlobs;

        params2.minThreshold = context->blobContext.minThreshold;
        params2.maxThreshold = context->blobContext.maxThreshold;
        params2.thresholdStep = context->blobContext.thresholdStep;
        params2.filterByArea = context->blobContext.filterByArea;
        params2.minArea = context->blobContext.minArea;
        params2.maxArea = context->blobContext.maxArea;
        params2.minDistBetweenBlobs = context->blobContext.minDistBetweenBlobs;

        cv::Ptr<cv::SimpleBlobDetector> detector1 = cv::SimpleBlobDetector::create(params1);
        cv::Ptr<cv::SimpleBlobDetector> detector2 = cv::SimpleBlobDetector::create(params2);
        detector1->detect(context->gray1, keypoints1);
        detector2->detect(context->gray2, keypoints2);
    }

    // Compute descriptors
    cv::Mat descriptors1, descriptors2;
    if (context->descriptorType == "sift") {
        computeSIFTDescriptors(context->gray1, keypoints1, descriptors1);
        computeSIFTDescriptors(context->gray2, keypoints2, descriptors2);
    } else if (context->descriptorType == "lbp") {
        computeLBPDescriptors(context->gray1, keypoints1, descriptors1);
        computeLBPDescriptors(context->gray2, keypoints2, descriptors2);
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

    // Visualize all matches from KNN
    std::vector<cv::DMatch> all_matches;
    for (const auto& knn_match : knn_matches) {
        if (!knn_match.empty()) {
            all_matches.push_back(knn_match[0]);
        }
    }

    // // Visualize good matches
    // // Apply Lowe's ratio test to filter good matches
    // const float ratio_thresh = context->ratioThreshold / 100.0f;
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
    cv::drawMatches(context->img1, keypoints1, context->img2, keypoints2, all_matches, imgMatches,
                    cv::Scalar::all(-1), cv::Scalar::all(-1),
                    std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

    // Resize the visualization to fit the screen
    cv::Mat imgMatchesResized;
    double scale = 0.5;
    cv::resize(imgMatches, imgMatchesResized, cv::Size(), scale, scale);

    cv::imshow("Feature Matches", imgMatchesResized);
}

void matchFeatures(const std::string& detectorType, const std::string& descriptorType, 
                  const std::string& img1Path, const std::string& img2Path) {
    MatchContext* context = new MatchContext();
    const std::string windowName = "Feature Matches";

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
    context->img1 = cv::imread(img1Path, cv::IMREAD_COLOR);
    context->img2 = cv::imread(img2Path, cv::IMREAD_COLOR);
    if (context->img1.empty() || context->img2.empty()) {
        std::cerr << "Error: Could not read one of the images." << std::endl;
        return;
    }

    cv::cvtColor(context->img1, context->gray1, cv::COLOR_BGR2GRAY);
    cv::cvtColor(context->img2, context->gray2, cv::COLOR_BGR2GRAY);

    context->detectorType = detectorType;
    context->descriptorType = descriptorType;

    // Create window and trackbar
    cv::namedWindow(windowName, cv::WINDOW_NORMAL);

    if (context->detectorType == "harris") {
        cv::createTrackbar("Block Size", windowName, &context->harrisContext.blockSize, context->harrisContext.max_harris_blockSize, onMatchTrackbar, context);
        cv::createTrackbar("Aperture (Odd)", windowName, &context->harrisContext.apertureSize, context->harrisContext.max_harris_ksize, onMatchTrackbar, context);
        cv::createTrackbar("K (x100)", windowName, &context->harrisContext.k_x100, context->harrisContext.max_harris_k_x100, onMatchTrackbar, context);
        cv::createTrackbar("Threshold", windowName, &context->harrisContext.threshold, context->harrisContext.max_harris_threshold, onMatchTrackbar, context);

    } else if (context->detectorType == "dog") {
        cv::createTrackbar("Sigma (first kernel)", windowName, &context->dogContext.sigma1, 100, onMatchTrackbar, context);
        cv::createTrackbar("Sigma Diff (first to second kernel)", windowName, &context->dogContext.sigmaDiff, 100, onMatchTrackbar, context);
        cv::createTrackbar("Kernel Size", windowName, &context->dogContext.kernelSize, 21, onMatchTrackbar, context);

    } else if (context->detectorType == "blob") {
        cv::createTrackbar("Min Threshold", windowName, &context->blobContext.minThreshold, 255, onMatchTrackbar, context);
        cv::createTrackbar("Max Threshold", windowName, &context->blobContext.maxThreshold, 255, onMatchTrackbar, context);
        cv::createTrackbar("Threshold Step", windowName, &context->blobContext.thresholdStep, 10, onMatchTrackbar, context);
    }

    cv::createTrackbar("Ratio Thresh (x100)", windowName, &context->ratioThreshold, 100, onMatchTrackbar, context);

    // Initial call to display matches
    onMatchTrackbar(0, context);

    cv::waitKey(0);
    cv::destroyAllWindows();
    delete context;
}

void matchFeaturesCamera(const std::string& detectorType, const std::string& descriptorType) {
    MatchContext* context = new MatchContext();
    const std::string windowName = "Feature Matches";

    std::cout << "Matching from camera" << std::endl;
    std::cout << "Detector: " << detectorType << ", Descriptor: " << descriptorType << std::endl;
    if (detectorType != "harris" && detectorType != "dog" && detectorType != "blob") {
        std::cerr << "Error: Unknown detector type '" << detectorType << "'" << std::endl;
        return;
    }

    if (descriptorType != "sift" && descriptorType != "lbp") {
        std::cerr << "Error: Unknown descriptor type '" << descriptorType << "'" << std::endl;
        return;
    }

    cv::VideoCapture cap(0);
    if (!cap.isOpened()) {
        std::cerr << "Error: Could not open camera." << std::endl;
        return;
    }

    context->detectorType = detectorType;
    context->descriptorType = descriptorType;

    // Create window and trackbar
    cv::namedWindow(windowName, cv::WINDOW_NORMAL);

    if (context->detectorType == "harris") {
        cv::createTrackbar("Block Size", windowName, &context->harrisContext.blockSize, context->harrisContext.max_harris_blockSize, onMatchTrackbar, context);
        cv::createTrackbar("Aperture (Odd)", windowName, &context->harrisContext.apertureSize, context->harrisContext.max_harris_ksize, onMatchTrackbar, context);
        cv::createTrackbar("K (x100)", windowName, &context->harrisContext.k_x100, context->harrisContext.max_harris_k_x100, onMatchTrackbar, context);
        cv::createTrackbar("Threshold", windowName, &context->harrisContext.threshold, context->harrisContext.max_harris_threshold, onMatchTrackbar, context);

    } else if (context->detectorType == "dog") {
        cv::createTrackbar("Sigma (first kernel)", windowName, &context->dogContext.sigma1, 100, onMatchTrackbar, context);
        cv::createTrackbar("Sigma Diff (first to second kernel)", windowName, &context->dogContext.sigmaDiff, 100, onMatchTrackbar, context);
        cv::createTrackbar("Kernel Size", windowName, &context->dogContext.kernelSize, 21, onMatchTrackbar, context);
    } else if (context->detectorType == "blob") {
        cv::createTrackbar("Min Threshold", windowName, &context->blobContext.minThreshold, 255, onMatchTrackbar, context);
        cv::createTrackbar("Max Threshold", windowName, &context->blobContext.maxThreshold, 255, onMatchTrackbar, context);
        cv::createTrackbar("Threshold Step", windowName, &context->blobContext.thresholdStep, 10, onMatchTrackbar, context);
    }      

    cv::createTrackbar("Ratio Thresh (x100)", windowName, &context->ratioThreshold, 100, onMatchTrackbar, context);
    
    cv::Mat frame;
    std::cout << "Press SPACE to capture first image..." << std::endl;
    while (true) {
        cap >> frame;
        if (frame.empty()) continue;
        cv::imshow("Capture Image 1 (Press SPACE to capture)", frame);
        int key = cv::waitKey(30);
        if (key == 32) { // SPACE key
            context->img1 = frame.clone();
            break;
        }
    }
    cv::destroyWindow("Capture Image 1 (Press SPACE to capture)");

    cv::cvtColor(context->img1, context->gray1, cv::COLOR_BGR2GRAY);

    std::cout << "Press SPACE to capture second image..." << std::endl;
    while (true) {
        cap >> frame;
        if (frame.empty()) continue;
        cv::imshow("Capture Image 2 (Press SPACE to capture)", frame);
        int key = cv::waitKey(30);
        if (key == 32) { // SPACE key
            context->img2 = frame.clone();
            break;
        }
    }
    cv::destroyWindow("Capture Image 2 (Press SPACE to capture)");

    cv::cvtColor(context->img2, context->gray2, cv::COLOR_BGR2GRAY);

    onMatchTrackbar(0, context);

    cv::waitKey(0);
    cv::destroyAllWindows();
    delete context;
}