#include "DoGDetector.hpp"
#include "Utils.hpp"
#include <iostream>

std::vector<cv::KeyPoint> getDoGKeypoints(const cv::Mat& dogResponse, float threshold) {
    std::vector<cv::KeyPoint> keypoints;
    for (int y = 0; y < dogResponse.rows; y++) {
        for (int x = 0; x < dogResponse.cols; x++) {
            float response = dogResponse.at<uchar>(y, x);
            if (response > threshold) {
                keypoints.push_back(cv::KeyPoint(cv::Point2f(x, y), 10.f, -1, response));
            }
        }
    }

    // Limit to 5000 strongest keypoints (for example)
    std::sort(keypoints.begin(), keypoints.end(),
              [](const cv::KeyPoint& a, const cv::KeyPoint& b) { return a.response > b.response; });
    if (keypoints.size() > 5000)
        keypoints.resize(5000);

    // Apply Non-Maximum Suppression to refine keypoints
    return keypointNMS(keypoints, 10.0); // 10 pixels minimum distance
}

void onDoGTrackbar(int, void* userData) {
    DoGContext* dogContext = static_cast<DoGContext*>(userData);
    if (dogContext->src.empty()) return;

    int ksize = dogContext->getOddKernelSize();
    int sigmaDiff = dogContext->getSafeSigmaDiff();

    // // Create Gaussian kernels
    // cv::Mat gauss1 = createGaussianFilter(ksize);
    // cv::Mat gauss2 = createGaussianFilter(ksize);

    // // Apply Gaussian blurs
    // cv::Mat blur1 = applyConvolution(dogContext->gray, gauss1);
    // cv::Mat blur2 = applyConvolution(dogContext->gray, gauss2);

    cv::Mat blur1, blur2;
    cv::GaussianBlur(dogContext->gray, blur1, cv::Size(ksize, ksize), dogContext->sigma1);
    cv::GaussianBlur(dogContext->gray, blur2, cv::Size(ksize, ksize), dogContext->sigma1 + sigmaDiff);

    // Compute DoG
    cv::Mat dog;
    cv::subtract(blur1, blur2, dog);
    cv::Mat dogNorm;
    cv::normalize(dog, dogNorm, 0, 255, cv::NORM_MINMAX, CV_8UC1);

    cv::Mat result = dogContext->src.clone();
    std::vector<cv::KeyPoint> kps = getDoGKeypoints(dogNorm, dogContext->threshold);
    cv::drawKeypoints(dogContext->src, kps, result, cv::Scalar(0,255,255), 
                      cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

    cv::imshow("DoG Detection", result);
    cv::imshow("DoG Response", dogNorm);
}

void createDoGTrackbars(const std::string& windowName, DoGContext* ctx, void (*onCallback)(int, void*)) {
    cv::createTrackbar("Sigma (first kernel)", windowName, &ctx->sigma1, 100, onCallback, ctx);
    cv::createTrackbar("Sigma Diff (first to second kernel)", windowName, &ctx->sigmaDiff, 100, onCallback, ctx);
    cv::createTrackbar("Kernel Size", windowName, &ctx->kernelSize, 21, onCallback, ctx);
    cv::createTrackbar("Threshold", windowName, &ctx->threshold, 255, onCallback, ctx);
}

void detectDoG(const std::string& imagePath) {
    DoGContext* dogContext = new DoGContext;

    std::string windowName1 = "DoG Detection";
    std::string windowName2 = "DoG Response";
    cv::namedWindow(windowName1, cv::WINDOW_GUI_EXPANDED);
    cv::namedWindow(windowName2, cv::WINDOW_GUI_EXPANDED);

    dogContext->src = cv::imread(imagePath, cv::IMREAD_COLOR);
    if (dogContext->src.empty()) { std::cerr << "Could not read image: " << imagePath << std::endl; return; }

    cv::cvtColor(dogContext->src, dogContext->gray, cv::COLOR_BGR2GRAY);

    createDoGTrackbars(windowName1, dogContext, onDoGTrackbar);

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

    std::string windowName1 = "DoG Detection";
    cv::namedWindow(windowName1, cv::WINDOW_GUI_EXPANDED);
    std::string windowName2 = "DoG Response";
    cv::namedWindow(windowName2, cv::WINDOW_GUI_EXPANDED);

    DoGContext* dogContext = new DoGContext();
    createDoGTrackbars(windowName1, dogContext, onDoGTrackbar);

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
