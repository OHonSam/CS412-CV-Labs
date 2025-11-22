#include "DoGDetector.hpp"
#include <iostream>

void onDoGTrackbar(int, void* userData) {
    DoGContext* dogContext = static_cast<DoGContext*>(userData);
    if (dogContext->src.empty()) return;

    // Ensure kernel size is odd and at least 3
    int ksize = dogContext->kernelSize;
    ksize = (ksize / 2) * 2 + 1;
    if (ksize < 3) ksize = 3;

    // Ensure sigma difference is at least 1
    int sigmaDiff = dogContext->sigmaDiff;
    if (sigmaDiff < 1) sigmaDiff = 1;

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

    cv::imshow("DoG Detection", dogNorm);
}

void detectDoG(const std::string& imagePath) {
    DoGContext* dogContext = new DoGContext;

    std::string windowName = "DoG Detection";
    cv::namedWindow(windowName, cv::WINDOW_AUTOSIZE);

    dogContext->src = cv::imread(imagePath, cv::IMREAD_COLOR);
    if (dogContext->src.empty()) { std::cerr << "Could not read image: " << imagePath << std::endl; return; }

    cv::cvtColor(dogContext->src, dogContext->gray, cv::COLOR_BGR2GRAY);

    cv::createTrackbar("Sigma (first kernel)", windowName, &dogContext->sigma1, 100, onDoGTrackbar, dogContext);
    cv::createTrackbar("Sigma Diff (first to second kernel)", windowName, &dogContext->sigmaDiff, 100, onDoGTrackbar, dogContext);
    cv::createTrackbar("Kernel Size", windowName, &dogContext->kernelSize, 21, onDoGTrackbar, dogContext);

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
    cv::createTrackbar("Sigma (first kernel)", windowName, &dogContext->sigma1, 100, onDoGTrackbar, dogContext);
    cv::createTrackbar("Sigma Diff (first to second kernel)", windowName, &dogContext->sigmaDiff, 100, onDoGTrackbar, dogContext);
    cv::createTrackbar("Kernel Size", windowName, &dogContext->kernelSize, 21, onDoGTrackbar, dogContext);

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
