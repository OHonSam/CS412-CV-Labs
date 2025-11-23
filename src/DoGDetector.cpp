#include "DoGDetector.hpp"
#include "Utils.hpp"
#include <iostream>

std::vector<cv::KeyPoint> DoGDetector::detect(const cv::Mat& gray) {
    context.gray = gray.clone();

    int ksize = context.params.getValidKernelSize();
    int sigmaDiff = context.params.getValidSigmaDiff();

    cv::Mat blur1, blur2;
    cv::GaussianBlur(context.gray, blur1, cv::Size(ksize, ksize), context.params.sigma1);
    cv::GaussianBlur(context.gray, blur2, cv::Size(ksize, ksize), context.params.sigma1 + sigmaDiff);

    // Compute DoG
    cv::Mat dog;
    cv::subtract(blur1, blur2, dog);
    cv::Mat dogNorm;
    cv::normalize(dog, dogNorm, 0, 255, cv::NORM_MINMAX, CV_8UC1);

    return getDoGKeypoints(dogNorm, context.params.threshold);
}

void DoGDetector::createTrackbars(const std::string& win, void* userdata) {
    DoGContext* ctx = static_cast<DoGContext*>(userdata);

    cv::createTrackbar("Sigma (first kernel)", win, &ctx->params.sigma1, DoGParams::MAX_SIGMA, onDoGTrackbar, ctx);
    cv::createTrackbar("Sigma Diff (first to second kernel)", win, &ctx->params.sigmaDiff, DoGParams::MAX_SIGMA_DIFF, onDoGTrackbar, ctx);
    cv::createTrackbar("Kernel Size", win, &ctx->params.kernelSize, DoGParams::MAX_KERNEL_SIZE, onDoGTrackbar, ctx);
}

std::vector<cv::KeyPoint> getDoGKeypoints(const cv::Mat& dogResponse, float threshold) {
    std::vector<cv::KeyPoint> keypoints;
    for (int y = 0; y < dogResponse.rows; y++) {
        for (int x = 0; x < dogResponse.cols; x++) {
            float response = dogResponse.at<uchar>(y, x);
            if (response > threshold) {
                keypoints.push_back(cv::KeyPoint(cv::Point2f(x, y), 5.f, -1, response));
            }
        }
    }

    // Apply Non-Maximum Suppression to refine keypoints
    return keypointNMS(keypoints, 10.0); // 10 pixels minimum distance
}

void onDoGTrackbar(int, void* userData) {
    DoGContext* dogContext = static_cast<DoGContext*>(userData);
    if (dogContext->src.empty()) return;

    int ksize = dogContext->params.getValidKernelSize();
    int sigmaDiff = dogContext->params.getValidSigmaDiff();

    // // Create Gaussian kernels
    // cv::Mat gauss1 = createGaussianFilter(ksize);
    // cv::Mat gauss2 = createGaussianFilter(ksize);

    // // Apply Gaussian blurs
    // cv::Mat blur1 = applyConvolution(dogContext->gray, gauss1);
    // cv::Mat blur2 = applyConvolution(dogContext->gray, gauss2);

    cv::Mat blur1, blur2;
    cv::GaussianBlur(dogContext->gray, blur1, cv::Size(ksize, ksize), dogContext->params.sigma1);
    cv::GaussianBlur(dogContext->gray, blur2, cv::Size(ksize, ksize), dogContext->params.sigma1 + sigmaDiff);

    // Compute DoG
    cv::Mat dog;
    cv::subtract(blur1, blur2, dog);
    cv::Mat dogNorm;
    cv::normalize(dog, dogNorm, 0, 255, cv::NORM_MINMAX, CV_8UC1);

    cv::Mat result = dogContext->src.clone();
    std::vector<cv::KeyPoint> kps = getDoGKeypoints(dogNorm, dogContext->params.threshold);
    cv::drawKeypoints(dogContext->src, kps, result, cv::Scalar(0,0,255), 
                      cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

    cv::imshow("DoG Detection", result);
    cv::imshow("DoG Response", dogNorm);
}

void detectDoG(const std::string& imagePath) {
    DoGContext* dogContext = new DoGContext;

    std::string windowName = "DoG Detection";
    cv::namedWindow(windowName, cv::WINDOW_AUTOSIZE);

    dogContext->src = cv::imread(imagePath, cv::IMREAD_COLOR);
    if (dogContext->src.empty()) { std::cerr << "Could not read image: " << imagePath << std::endl; return; }

    cv::cvtColor(dogContext->src, dogContext->gray, cv::COLOR_BGR2GRAY);

    cv::createTrackbar("Sigma (first kernel)", windowName, 
        &dogContext->params.sigma1, 
        DoGParams::MAX_SIGMA, 
        onDoGTrackbar, 
        dogContext
    );

    cv::createTrackbar("Sigma Diff (first to second kernel)", windowName, 
        &dogContext->params.sigmaDiff, 
        DoGParams::MAX_SIGMA_DIFF, 
        onDoGTrackbar, 
        dogContext
    );
    cv::createTrackbar("Kernel Size", windowName, 
        &dogContext->params.kernelSize, 
        DoGParams::MAX_KERNEL_SIZE, 
        onDoGTrackbar, 
        dogContext
    );

    onDoGTrackbar(0, dogContext);

    cv::waitKey(0);
    cv::destroyAllWindows();
}

void detectDoGCamera() {
    cv::VideoCapture cap(0);
    if (!cap.isOpened()) {
        std::cerr << "Error: Could not open camera" << std::endl;
        return;
    }

    std::string windowName = "DoG Detection";
    cv::namedWindow(windowName, cv::WINDOW_AUTOSIZE);

    DoGContext* dogContext = new DoGContext();
    cv::createTrackbar(
        "Sigma (first kernel)", windowName, 
        &dogContext->params.sigma1, 
        DoGParams::MAX_SIGMA, 
        onDoGTrackbar, 
        dogContext
    );

    cv::createTrackbar(
        "Sigma Diff (first to second kernel)", windowName, 
        &dogContext->params.sigmaDiff, 
        DoGParams::MAX_SIGMA_DIFF, 
        onDoGTrackbar, 
        dogContext
    );

    cv::createTrackbar(
        "Kernel Size", windowName, 
        &dogContext->params.kernelSize, 
        DoGParams::MAX_KERNEL_SIZE, 
        onDoGTrackbar, 
        dogContext
    );

    cv::Mat frame;
    while (true) {
        cap >> frame;
        if (frame.empty()) break;

        // 2. Update Context with the new frame
        dogContext->src = frame;
        cv::cvtColor(dogContext->src, dogContext->gray, cv::COLOR_BGR2GRAY);

        // 3. Process and Display
        onDoGTrackbar(0, dogContext);

        if (cv::waitKey(30) >= 27) break;
    }

    cap.release();
    cv::destroyAllWindows();
}
