#include "HarrisDetector.hpp" 
#include "Utils.hpp"
#include <iostream>

std::vector<cv::KeyPoint> HarrisDetector::extractKeypoints(const cv::Mat& response, 
                                                            float threshold) {
    return getHarrisKeypoints(response, threshold);
}

cv::Mat HarrisDetector::computeHarrisResponse(const cv::Mat& gray, 
                                               const HarrisParams& params) {
    cv::Mat response, normalized;
    
    cv::cornerHarris(gray, response, 
                     params.getValidBlockSize(), 
                     params.getValidApertureSize(), 
                     params.getK());
    
    // Normalize to 0-255 range for consistent thresholding
    cv::normalize(response, normalized, 0, 255, cv::NORM_MINMAX, CV_32F, cv::Mat());
    
    return normalized;
}

std::vector<cv::KeyPoint> HarrisDetector::detect(const cv::Mat& gray) {
    context.gray = gray.clone();
    context.params = HarrisParams(); // Use default parameters

    if (gray.channels() != 1) {
        cv::cvtColor(gray, context.gray, cv::COLOR_BGR2GRAY);
    }
    
    cv::Mat response = computeHarrisResponse(context.gray, context.params);
    return extractKeypoints(response, static_cast<float>(context.params.threshold));
}

void HarrisDetector::createTrackbars(const std::string& win, void* userdata) {
    HarrisContext* ctx = static_cast<HarrisContext*>(userdata);
    
    cv::createTrackbar("Block Size", win, 
                       &ctx->params.blockSize, 
                       HarrisParams::MAX_BLOCK_SIZE, 
                       onHarrisMatchingTrackbar, ctx);
                       
    cv::createTrackbar("Aperture (Odd)", win, 
                       &ctx->params.apertureSize, 
                       HarrisParams::MAX_APERTURE_SIZE, 
                       onHarrisMatchingTrackbar, ctx);
                       
    cv::createTrackbar("K (x100)", win, 
                       &ctx->params.k_x100, 
                       HarrisParams::MAX_K_VALUE, 
                       onHarrisMatchingTrackbar, ctx);
                       
    cv::createTrackbar("Threshold", win, 
                       &ctx->params.threshold, 
                       HarrisParams::MAX_THRESHOLD, 
                       onHarrisMatchingTrackbar, ctx);
}

void onHarrisMatchingTrackbar(int, void* userData) {
    HarrisContext* ctx = static_cast<HarrisContext*>(userData);
    // Update parameters based on trackbar positions
    ctx->params.blockSize = cv::getTrackbarPos("Block Size", "Feature Matches");
    ctx->params.apertureSize = cv::getTrackbarPos("Aperture (Odd)", "Feature Matches");
    ctx->params.k_x100 = cv::getTrackbarPos("K (x100)", "Feature Matches");
    ctx->params.threshold = cv::getTrackbarPos("Threshold", "Feature Matches");
}

void myCornerHarris(cv::Mat& srcGray, cv::Mat& dst, int blockSize, int apertureSize, double k) {
    blockSize = (blockSize / 2) * 2 + 1;  // Force odd (e.g., 2 -> 3)
    if (blockSize < 3) blockSize = 3;     // Minimum size 3
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
    // Weighted sum to see gradient changes in multiple directions instead of
    // gradient at a single pixel
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
                keypoints.push_back(cv::KeyPoint(cv::Point2f(x, y), 5.f, -1, response));
            }
        }
    }

    return keypointNMS(keypoints, 10.0);  // Apply NMS with a minimum distance of 10 pixels
}

void onHarrisTrackbar(int, void* userData) {
    HarrisContext* ctx = static_cast<HarrisContext*>(userData);

    if (ctx->src.channels() == 3)
        cv::cvtColor(ctx->src, ctx->gray, cv::COLOR_BGR2GRAY);

    if (ctx->gray.empty()) {
        std::cerr << "Error: Grayscale image is empty." << std::endl;
        return;
    }
    
    // Compute Harris response with validated parameters
    cv::Mat response, normalized;
    cv::cornerHarris(ctx->gray, response, 
                     ctx->params.getValidBlockSize(),
                     ctx->params.getValidApertureSize(),
                     ctx->params.getK());
    
    // Uncomment to use custom implementation:
    // myCornerHarris(ctx->gray, response, 
    //                ctx->params.getValidBlockSize(),
    //                ctx->params.getValidApertureSize(),
    //                ctx->params.getK());
    
    cv::normalize(response, normalized, 0, 255, cv::NORM_MINMAX, CV_32F, cv::Mat());
    
    // Extract and draw keypoints
    std::vector<cv::KeyPoint> keypoints = getHarrisKeypoints(
        normalized, 
        static_cast<float>(ctx->params.threshold)
    );
    
    cv::Mat result = ctx->src.clone();
    cv::drawKeypoints(result, keypoints, result, 
                      cv::Scalar(0, 0, 255), 
                      cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    
    cv::imshow("Harris Corners", result);
    
    // Display response heatmap
    cv::Mat heatmap;
    cv::convertScaleAbs(normalized, heatmap);
    cv::imshow("Harris Response", heatmap);
}

void detectHarris(const std::string& imagePath) {
    HarrisContext* ctx = new HarrisContext();
    ctx->src = cv::imread(imagePath, cv::IMREAD_COLOR);
    
    if (ctx->src.empty()) {
        std::cerr << "Error: Could not read image: " << imagePath << std::endl;
        delete ctx;
        return;
    }

    cv::cvtColor(ctx->src, ctx->gray, cv::COLOR_BGR2GRAY);

    const std::string windowName = "Harris Corners";
    cv::namedWindow(windowName, cv::WINDOW_AUTOSIZE);

    // Create interactive trackbars
    cv::createTrackbar("Block Size", windowName, 
                       &ctx->params.blockSize, 
                       HarrisParams::MAX_BLOCK_SIZE, 
                       onHarrisTrackbar, ctx);
                       
    cv::createTrackbar("Aperture (Odd)", windowName, 
                       &ctx->params.apertureSize, 
                       HarrisParams::MAX_APERTURE_SIZE, 
                       onHarrisTrackbar, ctx);
                       
    cv::createTrackbar("K (x100)", windowName, 
                       &ctx->params.k_x100, 
                       HarrisParams::MAX_K_VALUE, 
                       onHarrisTrackbar, ctx);
                       
    cv::createTrackbar("Threshold", windowName, 
                       &ctx->params.threshold, 
                       HarrisParams::MAX_THRESHOLD, 
                       onHarrisTrackbar, ctx);

    // Display initial result
    onHarrisTrackbar(0, ctx);

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

    const std::string windowName = "Harris Corners";
    cv::namedWindow(windowName, cv::WINDOW_AUTOSIZE);

    HarrisContext* ctx = new HarrisContext();

    // Create trackbars
    cv::createTrackbar("Block Size", windowName, 
                       &ctx->params.blockSize, 
                       HarrisParams::MAX_BLOCK_SIZE, 
                       onHarrisTrackbar, ctx);
                       
    cv::createTrackbar("Aperture (Odd)", windowName, 
                       &ctx->params.apertureSize, 
                       HarrisParams::MAX_APERTURE_SIZE, 
                       onHarrisTrackbar, ctx);
                       
    cv::createTrackbar("K (x100)", windowName, 
                       &ctx->params.k_x100, 
                       HarrisParams::MAX_K_VALUE, 
                       onHarrisTrackbar, ctx);
                       
    cv::createTrackbar("Threshold", windowName, 
                       &ctx->params.threshold, 
                       HarrisParams::MAX_THRESHOLD, 
                       onHarrisTrackbar, ctx);

    cv::Mat frame;
    while (true) {
        cap >> frame;
        if (frame.empty()) break;

        // Update context with new frame
        ctx->src = frame;
        cv::cvtColor(ctx->src, ctx->gray, cv::COLOR_BGR2GRAY);

        // Process and display
        onHarrisTrackbar(0, ctx);

        // Exit on ESC key
        if (cv::waitKey(30) == 27) break;
    }

    cap.release();
    cv::destroyAllWindows();
    delete ctx;
}