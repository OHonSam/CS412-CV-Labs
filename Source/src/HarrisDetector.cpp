#include "HarrisDetector.hpp" 
#include "Utils.hpp"
#include <iostream>

void myCornerHarris(cv::Mat& srcGray, cv::Mat& dst, int blockSize, int apertureSize, double k) {
    blockSize = (blockSize / 2) * 2 + 1; // Force odd (e.g., 2 -> 3)
    if (blockSize < 3) blockSize = 3;    // Minimum size 3
    dst = cv::Mat::zeros(srcGray.size(), CV_32FC1);

    // 1. Compute gradients Ix and Iy using Sobel operator
    cv::Mat Ix, Iy;
    cv::Sobel(srcGray, Ix, CV_32F, 1, 0, apertureSize);
    cv::Sobel(srcGray, Iy, CV_32F, 0, 1, apertureSize);

    // 2. Compute products of derivatives
    cv::Mat Ix2 = Ix.mul(Ix);
    cv::Mat Iy2 = Iy.mul(Iy);
    cv::Mat Ixy = Ix.mul(Iy);

    // 3. Apply Gaussian filter to the derivative products
    // Weighted sum to see gradient changes in multiple directions instead of gradient at a single pixel
    cv::Mat Sx2, Sy2, Sxy;
    cv::GaussianBlur(Ix2, Sx2, cv::Size(blockSize, blockSize), 2);
    cv::GaussianBlur(Iy2, Sy2, cv::Size(blockSize, blockSize), 2);
    cv::GaussianBlur(Ixy, Sxy, cv::Size(blockSize, blockSize), 2);

    // 4. Compute Harris Response R for every pixel
    // M = [ Sx2  Sxy ]
    //     [ Sxy  Sy2 ]
    // R = det(M) - k * (trace(M))^2
    // det(M) = Sx2 * Sy2 - Sxy^2
    // trace(M) = Sx2 + Sy2
    for (int y = 0; y < srcGray.rows; y++) {
        for (int x = 0; x < srcGray.cols; x++) {
            float Sx2_val = Sx2.at<float>(y, x);
            float Sy2_val = Sy2.at<float>(y, x);
            float Sxy_val = Sxy.at<float>(y, x);

            float detM = (Sx2_val * Sy2_val) - (Sxy_val * Sxy_val);
            float traceM = Sx2_val + Sy2_val;

            dst.at<float>(y, x) = detM - k * (traceM * traceM);
        }
    }
}

std::vector<cv::KeyPoint> getHarrisKeypoints(const cv::Mat& harrisResponse, float threshold) {
    std::vector<cv::KeyPoint> keypoints;
    for (int y = 0; y < harrisResponse.rows; y++) {
        for (int x = 0; x < harrisResponse.cols; x++) {
            float response = harrisResponse.at<float>(y, x);
            if (response > threshold) {
                keypoints.push_back(cv::KeyPoint(cv::Point2f(x, y), 10.f, -1, response));
            }
        }
    }

    // Limit to 5000 strongest keypoints
    std::sort(keypoints.begin(), keypoints.end(),
              [](const cv::KeyPoint& a, const cv::KeyPoint& b) { return a.response > b.response; });
    if (keypoints.size() > 5000)
        keypoints.resize(5000);
    
    return keypointNMS(keypoints, 10.0); // Apply NMS with a minimum distance of 10 pixels
}

void onHarrisTrackbar(int, void* userData) {
    HarrisContext* ctx = static_cast<HarrisContext*>(userData);

    // 1. Constraints
    int odd_aperture = ctx->getOddAperture();
    int safe_block = ctx->getSafeBlock();
    double k = ctx->getK();

    // 2. Run Harris
    cv::Mat dst, dst_norm;

    cv::cornerHarris(ctx->gray, dst, safe_block, odd_aperture, k);
    // myCornerHarris(ctx->gray, dst, safe_block, odd_aperture, k);

    // Normalize the result to 0-255 range
    cv::normalize(dst, dst_norm, 0, 255, cv::NORM_MINMAX, CV_32FC1, cv::Mat());

    // 3. Draw
    cv::Mat result = ctx->src.clone();
    std::vector<cv::KeyPoint> keypoints = getHarrisKeypoints(dst_norm, static_cast<float>(ctx->threshold));
    cv::drawKeypoints(ctx->src, keypoints, result, cv::Scalar(0, 255, 255), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    
    std::cout << "Harris Keypoints Detected: " << keypoints.size() << std::endl;

    cv::imshow("Harris Corners", result);

    // 4. Show heatmap
    cv::Mat dst_norm_scaled;
    cv::convertScaleAbs(dst_norm, dst_norm_scaled);
    cv::imshow("Harris Response", dst_norm_scaled);
}

void createHarrisTrackbars(const std::string& windowName, HarrisContext* ctx, void (*onCallback)(int, void*)) {
    cv::createTrackbar("Block Size", windowName, &ctx->blockSize, ctx->max_harris_blockSize, onCallback, ctx);
    cv::createTrackbar("Aperture (Odd)", windowName, &ctx->apertureSize, ctx->max_harris_ksize, onCallback, ctx);
    cv::createTrackbar("K (x100)", windowName, &ctx->k_x100, ctx->max_harris_k_x100, onCallback, ctx);
    cv::createTrackbar("Threshold", windowName, &ctx->threshold, ctx->max_harris_threshold, onCallback, ctx);
}

void detectHarris(const std::string& imagePath) {
    HarrisContext* ctx = new HarrisContext();
    ctx->src = cv::imread(imagePath, cv::IMREAD_COLOR);

    if (ctx->src.empty()) { std::cerr << "Could not read image: " << imagePath << std::endl; return; }

    cv::cvtColor(ctx->src, ctx->gray, cv::COLOR_BGR2GRAY);

    std::string windowName1 = "Harris Corners";
    cv::namedWindow(windowName1, cv::WINDOW_NORMAL);
    std::string windowName2 = "Harris Response";
    cv::namedWindow(windowName2, cv::WINDOW_NORMAL);

    // Create trackbars to adjust parameters
    createHarrisTrackbars(windowName1, ctx, onHarrisTrackbar);

    // Initial call to display corners
    onHarrisTrackbar(0, ctx);
    // When moving the sliders, on_harris_trackbar(_, ctx) will be called automatically

    cv::waitKey(0);
    cv::destroyAllWindows();

    delete ctx;
}

void detectHarrisCamera() {
    cv::VideoCapture cap(0);
    if (!cap.isOpened()) {
        std::cerr << "Error: Could not open camera" << std::endl;
        return;
    }

    std::string windowName = "Harris Corners";
    cv::namedWindow(windowName, cv::WINDOW_AUTOSIZE);

    HarrisContext* ctx = new HarrisContext();

    cv::createTrackbar("Block Size", windowName, &ctx->blockSize, ctx->max_harris_blockSize, onHarrisTrackbar, ctx);
    cv::createTrackbar("Aperture (Odd)", windowName, &ctx->apertureSize, ctx->max_harris_ksize, onHarrisTrackbar, ctx);
    cv::createTrackbar("K (x100)", windowName, &ctx->k_x100, ctx->max_harris_k_x100, onHarrisTrackbar, ctx);
    cv::createTrackbar("Threshold", windowName, &ctx->threshold, ctx->max_harris_threshold, onHarrisTrackbar, ctx);

    cv::Mat frame;
    while (true) {
        cap >> frame;
        if (frame.empty()) break;

        // 2. Update Context with the new frame
        ctx->src = frame;
        cv::cvtColor(ctx->src, ctx->gray, cv::COLOR_BGR2GRAY);

        // 3. Process and Display
        // Call the trackbar callback to run detection on the new frame
        onHarrisTrackbar(0, ctx);

        if (cv::waitKey(30) == 27) break;
    }

    cap.release();
    cv::destroyAllWindows();
    delete ctx;
}