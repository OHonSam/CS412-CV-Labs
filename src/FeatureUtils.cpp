#include "FeatureUtils.hpp" 
#include <iostream>

void displayHelp() {
    std::cout << "========================================\n" << std::endl;

    std::cout << "USAGE:" << std::endl;
    std::cout << "  program.exe <command> [arguments]\n" << std::endl;

    std::cout << "COMMANDS:" << std::endl;
    std::cout << "  (a) harris <image.jpg>" << std::endl;
    std::cout << "      - Detect keypoints using Harris Corner detector\n" << std::endl;

    std::cout << "  (b) blob <image.jpg>" << std::endl;
    std::cout << "      - Detect keypoints using Blob detector\n" << std::endl;

    std::cout << "  (c) dog <image.jpg>" << std::endl;
    std::cout << "      - Detect keypoints using DoG (SIFT)\n" << std::endl;

    std::cout << "  (d) m harris sift <image1.jpg> <image2.jpg>" << std::endl;
    std::cout << "      - Match images using Harris + SIFT\n" << std::endl;

    std::cout << "  (e) m dog sift <image1.jpg> <image2.jpg>" << std::endl;
    std::cout << "      - Match images using DoG + SIFT\n" << std::endl;

    std::cout << "  (f) m blob sift <image1.jpg> <image2.jpg>" << std::endl;
    std::cout << "      - Match images using Blob + SIFT\n" << std::endl;

    std::cout << "  (g) m harris lbp <image1.jpg> <image2.jpg>" << std::endl;
    std::cout << "      - Match images using Harris + LBP\n" << std::endl;

    std::cout << "  (h) m dog lbp <image1.jpg> <image2.jpg>" << std::endl;
    std::cout << "      - Match images using DoG + LBP\n" << std::endl;

    std::cout << "  (i) m blob lbp <image1.jpg> <image2.jpg>" << std::endl;
    std::cout << "      - Match images using Blob + LBP\n" << std::endl;

    std::cout << "  (j) help" << std::endl;
    std::cout << "      - Display this help message\n" << std::endl;

    std::cout << "KEYBOARD CONTROLS:" << std::endl;
    std::cout << "  'esc' - Quit Video" << std::endl;
    std::cout << "  'h' - Display help" << std::endl;
    std::cout << "========================================\n" << std::endl;
}

void openCamera() {
    cv::VideoCapture cap(0);
    if (!cap.isOpened()) {
        std::cerr << "Error: Could not open camera" << std::endl;
        return;
    }

    cv::Mat frame;
    cv::namedWindow("Live Video Feed", cv::WINDOW_AUTOSIZE);

    while (true) {
        cap >> frame;
        if (frame.empty()) break;

        cv::imshow("Live Video Feed", frame);
        if (cv::waitKey(30) == 27) break; // ESC to quit
    }
    cap.release();
    cv::destroyAllWindows();
}

void onHarrisTrackbar(int, void* userData) {
    HarrisContext* ctx = static_cast<HarrisContext*>(userData);

    // 1. Constraints
    // Aperture size must be odd (1, 3, 5, 7)
    int odd_aperture = (ctx->apertureSize / 2) * 2 + 1;
    if(odd_aperture > 7) odd_aperture = 7;
    
    // Block size must be at least 2
    int safe_block = std::max(2, ctx->blockSize);

    double k = ctx->k_x100 / 100.0;

    // 2. Run Harris
    cv::Mat dst, dst_norm;

    cv::cornerHarris(ctx->gray, dst, safe_block, odd_aperture, k);
    // Normalize the result to 0-255 range
    cv::normalize(dst, dst_norm, 0, 255, cv::NORM_MINMAX, CV_32FC1, cv::Mat());

    // 3. Draw
    cv::Mat result = ctx->src.clone();
    for(int j = 0; j < dst_norm.rows; j++) {
        for(int i = 0; i < dst_norm.cols; i++) {
            if((int)dst_norm.at<float>(j,i) > ctx->threshold) {
                cv::circle(result, cv::Point(i,j), 5, cv::Scalar(0,0,255), 2);
            }
        }
    }
    cv::imshow("Harris Corners", result);

    // 4. Show heatmap
    cv::Mat dst_norm_scaled;
    cv::convertScaleAbs(dst_norm, dst_norm_scaled);
    cv::imshow("Harris Response", dst_norm_scaled);
}

void detectHarris(const std::string& imagePath) {
    HarrisContext* ctx = new HarrisContext();
    ctx->src = cv::imread(imagePath, cv::IMREAD_COLOR);
    if (ctx->src.empty()) { std::cerr << "Could not read image: " << imagePath << std::endl; return; }

    cv::cvtColor(ctx->src, ctx->gray, cv::COLOR_BGR2GRAY);

    std::string windowName = "Harris Corners";
    cv::namedWindow(windowName, cv::WINDOW_AUTOSIZE);

    // Create trackbars to adjust parameters
    cv::createTrackbar("Block Size", windowName, &ctx->blockSize, ctx->max_harris_blockSize, onHarrisTrackbar, ctx);
    cv::createTrackbar("Aperture (Odd)", windowName, &ctx->apertureSize, ctx->max_harris_ksize, onHarrisTrackbar, ctx);
    cv::createTrackbar("K (x100)", windowName, &ctx->k_x100, ctx->max_harris_k_x100, onHarrisTrackbar, ctx);
    cv::createTrackbar("Threshold", windowName, &ctx->threshold, ctx->max_harris_threshold, onHarrisTrackbar, ctx);

    // Initial call to display corners
    onHarrisTrackbar(0, ctx);
    // When moving the sliders, on_harris_trackbar(_, ctx) will be called automatically

    cv::waitKey(0);
    cv::destroyAllWindows();

    delete ctx;
}

void detectBlob(const std::string& imagePath) {
    cv::Mat img = cv::imread(imagePath, cv::IMREAD_COLOR);
    if (img.empty()) { std::cerr << "Could not read image: " << imagePath << std::endl; return; }

    // TODO: Implement Blob logic here
    cv::imshow("Blob", img);
    cv::waitKey(0);
    cv::destroyAllWindows();
}

void detectDoG(const std::string& imagePath) {
    cv::Mat img = cv::imread(imagePath, cv::IMREAD_COLOR);
    if (img.empty()) { std::cerr << "Could not read image: " << imagePath << std::endl; return; }

    // TODO: Implement DoG logic here
    cv::imshow("DoG", img);
    cv::waitKey(0);
    cv::destroyAllWindows();
}

void matchFeatures(const std::string& detectorType, const std::string& descriptorType, 
                  const std::string& img1Path, const std::string& img2Path) {
    std::cout << "Matching " << img1Path << " and " << img2Path << std::endl;
    std::cout << "Detector: " << detectorType << ", Descriptor: " << descriptorType << std::endl;
    // TODO: Implement matching logic
}