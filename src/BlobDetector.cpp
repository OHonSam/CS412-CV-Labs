#include "BlobDetector.hpp" 
#include <iostream>

std::vector<cv::KeyPoint> myBlobDetection(const cv::Mat& gray, const BlobContext& context) {
    // Step 1 & 2: Threshold at multiple levels and collect centers
    std::vector<cv::Point2f> allCenters; // Store all centers from all thresholds
    
    for (float thresh = context.minThreshold;
            thresh < context.maxThreshold;
            thresh += context.thresholdStep) {

        cv::Mat binaryImage;
        cv::threshold(gray, binaryImage, thresh, 255, cv::THRESH_BINARY);

        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(binaryImage, contours, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);
        
        for (const auto& contour : contours) {
            double area = cv::contourArea(contour);
            if (area >= context.minArea && area <= context.maxArea) {
                cv::Moments M = cv::moments(contour);
                if (M.m00 != 0) {
                    float cx = static_cast<float>(M.m10 / M.m00);
                    float cy = static_cast<float>(M.m01 / M.m00);
                    allCenters.push_back(cv::Point2f(cx, cy));
                }
            }
        }
    }

    // Step 3: Group close centers using minDistBetweenBlobs
    std::vector<cv::KeyPoint> keypoints;
    std::vector<bool> used(allCenters.size(), false);
    
    float minDist = context.minDistBetweenBlobs;
    
    for (size_t i = 0; i < allCenters.size(); i++) {
        if (used[i]) continue;
        
        // Start a new group
        std::vector<cv::Point2f> group;
        group.push_back(allCenters[i]);
        used[i] = true;
        
        // Find all centers within minDist
        for (size_t j = i + 1; j < allCenters.size(); j++) {
            if (used[j]) continue;
            
            float dist = cv::norm(allCenters[i] - allCenters[j]);
            if (dist < minDist) {
                group.push_back(allCenters[j]);
                used[j] = true;
            }
        }
        
        // Step 4: Compute final center and radius from the group
        cv::Point2f finalCenter(0, 0);
        for (const auto& pt : group) {
            finalCenter += pt;
        }
        finalCenter.x /= group.size();
        finalCenter.y /= group.size();
        
        // Estimate radius as average distance from center
        float avgRadius = 0;
        for (const auto& pt : group) {
            avgRadius += cv::norm(pt - finalCenter);
        }
        avgRadius /= group.size();
        
        // Only add if radius is reasonable (otherwise it's noise)
        if (avgRadius > 1.0f) {
            keypoints.emplace_back(finalCenter, avgRadius * 2.0f); // Diameter
        }
    }

    return keypoints;
}

void onBlobTrackbar(int, void* userData) {
    BlobContext* blobContext = static_cast<BlobContext*>(userData);
    if (blobContext->src.empty()) return;

    cv::SimpleBlobDetector::Params blobParams;

    blobParams.filterByArea = blobContext->filterByArea;
    blobParams.filterByCircularity = blobContext->filterByCircularity;
    blobParams.filterByConvexity = blobContext->filterByConvexity;
    blobParams.filterByInertia = blobContext->filterByInertia;
    blobParams.minThreshold = blobContext->minThreshold;
    blobParams.maxThreshold = blobContext->maxThreshold;
    blobParams.thresholdStep = blobContext->getSafeThresholdStep();
    blobParams.minArea = blobContext->minArea;
    blobParams.maxArea = blobContext->maxArea;
    blobParams.minCircularity = blobContext->getMinCircularity();
    blobParams.minConvexity = blobContext->getMinConvexity();
    blobParams.minInertiaRatio = blobContext->getMinInertia();

    cv::Ptr<cv::SimpleBlobDetector> detector = cv::SimpleBlobDetector::create(blobParams);
    std::vector<cv::KeyPoint> keypoints;
    detector->detect(blobContext->gray, keypoints);

    cv::Mat result = blobContext->src.clone();
    cv::drawKeypoints(blobContext->src, keypoints, result, cv::Scalar(0, 0, 255), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

    // // Own implementation start
    // std::vector<cv::KeyPoint> keypoints = myBlobDetection(blobContext->gray, *blobContext);

    // cv::Mat result;
    // cv::drawKeypoints(blobContext->src, keypoints, result, cv::Scalar(0, 0, 255), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    // // Own implementation end

    cv::imshow("Blob Detection", result);
}

void createBlobTrackbars(const std::string& windowName, BlobContext* ctx, void (*onCallback)(int, void*)) {
    cv::createTrackbar("Min Threshold", windowName, &ctx->minThreshold, 255, onCallback, ctx);
    cv::createTrackbar("Max Threshold", windowName, &ctx->maxThreshold, 255, onCallback, ctx);
    cv::createTrackbar("Threshold Step", windowName, &ctx->thresholdStep, 10, onCallback, ctx);
    cv::createTrackbar("Min Circularity", windowName, &ctx->minCircularity, 100, onCallback, ctx);
    cv::createTrackbar("Min Convexity", windowName, &ctx->minConvexity, 100, onCallback, ctx);
    cv::createTrackbar("Min Inertia", windowName, &ctx->minInertia, 100, onCallback, ctx);
}

void detectBlob(const std::string& imagePath) {
    BlobContext* blobContext = new BlobContext;

    std::string windowName = "Blob Detection";
    cv::namedWindow(windowName, cv::WINDOW_AUTOSIZE);

    blobContext->src = cv::imread(imagePath, cv::IMREAD_COLOR);
    if (blobContext->src.empty()) { std::cerr << "Could not read image: " << imagePath << std::endl; return; }

    cv::cvtColor(blobContext->src, blobContext->gray, cv::COLOR_BGR2GRAY);

    createBlobTrackbars(windowName, blobContext, onBlobTrackbar);

    onBlobTrackbar(0, blobContext);

    cv::waitKey(0);
    cv::destroyAllWindows();
    delete blobContext;
}

void detectBlobCamera() {
    cv::VideoCapture cap(0);
    if (!cap.isOpened()) {
        std::cerr << "Error: Could not open camera" << std::endl;
        return;
    }
    
    std::string windowName = "Blob Detection";
    cv::namedWindow(windowName, cv::WINDOW_AUTOSIZE);

    BlobContext* blobContext = new BlobContext();
    createBlobTrackbars(windowName, blobContext, onBlobTrackbar);

    cv::Mat frame;
    while (true) {
        cap >> frame;
        if (frame.empty()) break;

        // 2. Update Context with the new frame
        blobContext->src = frame;
        cv::cvtColor(blobContext->src, blobContext->gray, cv::COLOR_BGR2GRAY);

        // 3. Process and Display
        onBlobTrackbar(0, blobContext);

        if (cv::waitKey(30) >= 27) break;
    }

    cap.release();
    cv::destroyAllWindows();
}
