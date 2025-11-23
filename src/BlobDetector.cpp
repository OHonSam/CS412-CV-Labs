#include "BlobDetector.hpp" 
#include <iostream>

std::vector<cv::KeyPoint> BlobDetector::detect(const cv::Mat& gray) {
    context.gray = gray.clone();

    // Use SimpleBlobDetector with current context parameters
    cv::SimpleBlobDetector::Params blobParams;
    blobParams.filterByArea = context.params.filterByArea;
    blobParams.minArea = context.params.minArea;
    blobParams.maxArea = context.params.maxArea;
    blobParams.filterByCircularity = context.params.filterByCircularity;
    blobParams.filterByConvexity = context.params.filterByConvexity;
    blobParams.filterByInertia = context.params.filterByInertia;
    blobParams.minThreshold = context.params.minThreshold;
    blobParams.maxThreshold = context.params.maxThreshold;
    blobParams.thresholdStep = context.params.thresholdStep;

    cv::Ptr<cv::SimpleBlobDetector> detector = cv::SimpleBlobDetector::create(blobParams);
    std::vector<cv::KeyPoint> keypoints;
    detector->detect(context.gray, keypoints);
    
    return keypoints;
}

void BlobDetector::createTrackbars(const std::string& win, void* userdata) {
    BlobContext* ctx = static_cast<BlobContext*>(userdata);

    cv::createTrackbar("Min Threshold", win, &ctx->params.minThreshold, BlobParams::MAX_THRESHOLD, onBlobTrackbar, ctx);
    cv::createTrackbar("Max Threshold", win, &ctx->params.maxThreshold, BlobParams::MAX_THRESHOLD, onBlobTrackbar, ctx);
    cv::createTrackbar("Threshold Step", win, &ctx->params.thresholdStep, BlobParams::MAX_THRESHOLD_STEP, onBlobTrackbar, ctx);
    cv::createTrackbar("Min Area", win, &ctx->params.minArea, BlobParams::MAX_AREA, onBlobTrackbar, ctx);
}

std::vector<cv::KeyPoint> myBlobDetection(const cv::Mat& gray, const BlobContext& context) {
    // Step 1 & 2: Threshold at multiple levels and collect centers
    std::vector<cv::Point2f> allCenters; // Store all centers from all thresholds

    for (float thresh = context.params.minThreshold;
            thresh < context.params.maxThreshold;
            thresh += context.params.thresholdStep) {

        cv::Mat binaryImage;
        cv::threshold(gray, binaryImage, thresh, 255, cv::THRESH_BINARY);

        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(binaryImage, contours, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);
        
        for (const auto& contour : contours) {
            double area = cv::contourArea(contour);
            if (area >= context.params.minArea && area <= context.params.maxArea) {
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

    float minDist = context.params.minDistBetweenBlobs;

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

    blobParams.filterByArea = blobContext->params.filterByArea;
    blobParams.filterByCircularity = blobContext->params.filterByCircularity;
    blobParams.filterByConvexity = blobContext->params.filterByConvexity;
    blobParams.filterByInertia = blobContext->params.filterByInertia;
    blobParams.minThreshold = blobContext->params.minThreshold;
    blobParams.maxThreshold = blobContext->params.maxThreshold;
    blobParams.thresholdStep = blobContext->params.thresholdStep;

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

void detectBlob(const std::string& imagePath) {
    BlobContext* blobContext = new BlobContext;

    std::string windowName = "Blob Detection";
    cv::namedWindow(windowName, cv::WINDOW_AUTOSIZE);

    blobContext->src = cv::imread(imagePath, cv::IMREAD_COLOR);
    if (blobContext->src.empty()) { std::cerr << "Could not read image: " << imagePath << std::endl; return; }

    cv::cvtColor(blobContext->src, blobContext->gray, cv::COLOR_BGR2GRAY);

    cv::createTrackbar("Min Threshold", windowName, &blobContext->params.minThreshold, BlobParams::MAX_THRESHOLD, onBlobTrackbar, blobContext);
    cv::createTrackbar("Max Threshold", windowName, &blobContext->params.maxThreshold, BlobParams::MAX_THRESHOLD, onBlobTrackbar, blobContext);
    cv::createTrackbar("Threshold Step", windowName, &blobContext->params.thresholdStep, BlobParams::MAX_THRESHOLD_STEP, onBlobTrackbar, blobContext);
    cv::createTrackbar("Min Area", windowName, &blobContext->params.minArea, BlobParams::MAX_AREA, onBlobTrackbar, blobContext);

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

    cv::createTrackbar("Min Threshold", windowName, &blobContext->params.minThreshold, BlobParams::MAX_THRESHOLD, onBlobTrackbar, blobContext);
    cv::createTrackbar("Max Threshold", windowName, &blobContext->params.maxThreshold, BlobParams::MAX_THRESHOLD, onBlobTrackbar, blobContext);
    cv::createTrackbar("Threshold Step", windowName, &blobContext->params.thresholdStep, BlobParams::MAX_THRESHOLD_STEP, onBlobTrackbar, blobContext);
    cv::createTrackbar("Min Area", windowName, &blobContext->params.minArea, BlobParams::MAX_AREA, onBlobTrackbar, blobContext);

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
