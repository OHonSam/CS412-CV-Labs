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

    // cv::cornerHarris(ctx->gray, dst, safe_block, odd_aperture, k);
    myCornerHarris(ctx->gray, dst, safe_block, odd_aperture, k);

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

    // cv::SimpleBlobDetector::Params blobParams;

    // blobParams.filterByArea = blobContext->filterByArea;
    // blobParams.filterByCircularity = blobContext->filterByCircularity;
    // blobParams.filterByConvexity = blobContext->filterByConvexity;
    // blobParams.filterByInertia = blobContext->filterByInertia;
    // blobParams.minThreshold = blobContext->minThreshold;
    // blobParams.maxThreshold = blobContext->maxThreshold;
    // blobParams.thresholdStep = blobContext->thresholdStep;

    // cv::Ptr<cv::SimpleBlobDetector> detector = cv::SimpleBlobDetector::create(blobParams);
    // std::vector<cv::KeyPoint> keypoints;
    // detector->detect(blobContext->gray, keypoints);

    // cv::Mat result = blobContext->src.clone();
    // cv::drawKeypoints(blobContext->src, keypoints, result, cv::Scalar(0, 0, 255), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

    // Own implementation start
    std::vector<cv::KeyPoint> keypoints = myBlobDetection(blobContext->gray, *blobContext);

    cv::Mat result;
    cv::drawKeypoints(blobContext->src, keypoints, result, cv::Scalar(0, 0, 255), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    // Own implementation end

    cv::imshow("Blob Detection", result);
}

void detectBlob(const std::string& imagePath) {
    BlobContext* blobContext = new BlobContext;

    std::string windowName = "Blob Detection";
    cv::namedWindow(windowName, cv::WINDOW_AUTOSIZE);

    blobContext->src = cv::imread(imagePath, cv::IMREAD_COLOR);
    if (blobContext->src.empty()) { std::cerr << "Could not read image: " << imagePath << std::endl; return; }

    cv::cvtColor(blobContext->src, blobContext->gray, cv::COLOR_BGR2GRAY);

    cv::createTrackbar("Min Threshold", windowName, &blobContext->minThreshold, 255, onBlobTrackbar, blobContext);
    cv::createTrackbar("Max Threshold", windowName, &blobContext->maxThreshold, 255, onBlobTrackbar, blobContext);
    cv::createTrackbar("Threshold Step", windowName, &blobContext->thresholdStep, 10, onBlobTrackbar, blobContext);

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

    cv::createTrackbar("Min Threshold", windowName, &blobContext->minThreshold, 255, onBlobTrackbar, blobContext);
    cv::createTrackbar("Max Threshold", windowName, &blobContext->maxThreshold, 255, onBlobTrackbar, blobContext);
    cv::createTrackbar("Threshold Step", windowName, &blobContext->thresholdStep, 10, onBlobTrackbar, blobContext);

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

void matchFeatures(const std::string& detectorType, const std::string& descriptorType, 
                  const std::string& img1Path, const std::string& img2Path) {
    std::cout << "Matching " << img1Path << " and " << img2Path << std::endl;
    std::cout << "Detector: " << detectorType << ", Descriptor: " << descriptorType << std::endl;
    // TODO: Implement matching logic
}

cv::Mat createGaussianFilter(int size) { 
    // https://docs.opencv.org/2.4/modules/imgproc/doc/filtering.html#getgaussiankernel 
    float sigma = 0.3 * ((size - 1) * 0.5 -1) + 0.8;
    float alpha = 0.0f;
    int centerIdx = (size - 1) / 2;

    cv::Mat kernel = cv::Mat(size, size, CV_32F, cv::Scalar(1));

    for (int row = 0; row < size; ++row) { 
        for (int col = 0; col < size; ++col) { 
            int dy = row - centerIdx;
            int dx = col - centerIdx;

            // https://en.wikipedia.org/wiki/Gaussian_filter 
            // https://en.wikipedia.org/wiki/Gaussian_blur 
            
            float value = exp(-(dy*dy + dx*dx)/(2.0f * sigma * sigma));
            kernel.at<float>(row, col) = value;
            alpha += value;
        }
    } 
    
    // Normalize 
    for (int row = 0; row < size; ++row) { 
        for (int col = 0; col < size; ++col) { 
            kernel.at<float>(row, col) /= alpha;
        } 
    } 
    return kernel;
} 

cv::Mat applyConvolution(const cv::Mat& src, const cv::Mat& kernel) { 
    int imRows = src.rows;
    int imCols = src.cols;
    
    int kerRows = kernel.rows;
    int kerCols = kernel.cols;
    int kerCenterY = (kerRows - 1)/2;
    int kerCenterX = (kerCols - 1)/2;

    cv::Mat dst = cv::Mat(imRows, imCols, src.type(), cv::Scalar(1));

    for (int imRow = 0; imRow < imRows; ++imRow) { 
        for (int imCol = 0; imCol < imCols; ++imCol) { 
            // Use zero paddings so [col][row] will be at the center of kernel 
            if (src.channels() == 1) { 
                float value = 0.0f;

                for (int kerRow = 0; kerRow < kerRows; ++kerRow) { 
                    for (int kerCol = 0; kerCol < kerCols; ++kerCol) { 
                        int i = imRow + (kerRow - kerCenterY);
                        int j = imCol + (kerCol - kerCenterX);
                        
                        if (i < 0 || i >= imRows || j < 0 || j >= imCols) { 
                            continue;
                        } 
                        
                        value += src.at<uchar>(i, j) * kernel.at<float>(kerRow, kerCol);
                    } 
                } 
                
                // dst.at<float>(imRow, imCol) = value;

                // Clamp to valid range [0, 255]. Working with standard images (imread/imwrite) Displaying results with imshow() 
                dst.at<uchar>(imRow, imCol) = cv::saturate_cast<uchar>(value);
            } 
            else if (src.channels() == 3) { 
                float valueB = 0.0f, valueG =0.0f, valueR = 0.0f;
                
                for (int kerRow = 0; kerRow < kerRows; ++kerRow) { 
                    for (int kerCol = 0; kerCol < kerCols; ++kerCol) { 
                        int i = imRow + (kerRow - kerCenterY);
                        int j = imCol + (kerCol - kerCenterX);

                        if (i < 0 || i >= imRows || j < 0 || j >= imCols) { 
                            continue;
                        } 
                        
                        float kerValue = kernel.at<float>(kerRow, kerCol);

                        cv::Vec3b imPixel = src.at<cv::Vec3b>(i, j);
                        valueB += imPixel[0] * kerValue;
                        valueG += imPixel[1] * kerValue;
                        valueR += imPixel[2] * kerValue;
                    } 
                } 
                
                // dst.at<Vec3b>(imRow, imCol) =Vec3b(valueB, valueG, valueR);

                // Clamp to valid range [0, 255] 
                dst.at<cv::Vec3b>(imRow, imCol) = cv::Vec3b( 
                    cv::saturate_cast<uchar>(valueB), 
                    cv::saturate_cast<uchar>(valueG), 
                    cv::saturate_cast<uchar>(valueR) 
                );
            } 
        } 
    } 
    
    return dst;
} 

std::vector<float> createAverageFilter1D(int size) { 
    return std::vector<float>(size, 1.0f / size);
} 

std::vector<float> createGaussianFilter1D(int size) { 
    float sigma = 0.3 * ((size - 1) * 0.5 -1) + 0.8;
    float alpha = 0.0f;
    int centerIdx = (size - 1) / 2;

    std::vector<float> kernel(size);

    for (int i = 0; i < size; ++i) { 
        int d = i - centerIdx;
        float value = exp(-(d * d) / (2.0f * sigma * sigma));
        kernel[i] = value;
        alpha += value;
    } 
    
    for (int i = 0; i < size; ++i) { 
        kernel[i] /= alpha;
    } 
    
    return kernel;
} 

cv::Mat applyVerticalConvolution1D(const cv::Mat& src, const std::vector<float>& verticalKernel1D) { 
    int imRows = src.rows;
    int imCols = src.cols;

    int kerRows = verticalKernel1D.size();
    int kerCenterY = (kerRows - 1)/2;

    cv::Mat dst = cv::Mat(imRows, imCols, src.type(), cv::Scalar(1));

    for (int imRow = 0; imRow < imRows; ++imRow) { 
        for (int imCol = 0; imCol < imCols; ++imCol) { 
            // Use zero paddings so [col][row] will be at the center of kernel 
            if (src.channels() == 1) { 
                float value = 0.0f;

                for (int kerRow = 0; kerRow < kerRows; ++kerRow) { 
                    int i = imRow + (kerRow - kerCenterY);

                    if (i < 0 || i >= imRows) { 
                        continue;
                    } 
                    
                    value += src.at<uchar>(i, imCol) * verticalKernel1D[kerRow];
                } 
                
                dst.at<float>(imRow, imCol) = value;
            } 
            else if (src.channels() == 3) { 
                float valueB = 0.0f, valueG =0.0f, valueR = 0.0f;

                for (int kerRow = 0; kerRow < kerRows; ++kerRow) { 
                    int i = imRow + (kerRow - kerCenterY);
                    if (i < 0 || i >= imRows) { 
                        continue;
                    } 
                    
                    float kerValue = verticalKernel1D[kerRow];

                    cv::Vec3b imPixel = src.at<cv::Vec3b>(i, imCol);
                    valueB += imPixel[0] * kerValue;
                    valueG += imPixel[1] * kerValue;
                    valueR += imPixel[2] * kerValue;

                } 

                dst.at<cv::Vec3b>(imRow, imCol) = cv::Vec3b(valueB, valueG, valueR);
            } 
        } 
    } 
    return dst;
} 

cv::Mat applyHorizontalConvolution1D(const cv::Mat& src, const std::vector<float>& horizontalKernel1D) { 
    int imRows = src.rows;
    int imCols = src.cols;
    int kerCols = horizontalKernel1D.size();
    int kerCenterX = (kerCols - 1)/2;

    cv::Mat dst = cv::Mat(imRows, imCols, src.type(), cv::Scalar(1));

    for (int imRow = 0; imRow < imRows; ++imRow) { 
        for (int imCol = 0; imCol < imCols; ++imCol) { 
            // Use zero paddings so [col][row] will be at the center of kernel 
            if (src.channels() == 1) { 
                float value = 0.0f;

                for (int kerCol = 0; kerCol < kerCols; ++kerCol) { 
                    int j = imCol + (kerCol - kerCenterX);


                    if (j < 0 || j >= imCols) { 
                        continue;
                    } 
                    
                    value += src.at<uchar>(imRow, j) * horizontalKernel1D[kerCol];
                } 
                
                dst.at<float>(imRow, imCol) = value;

            } 
            else if (src.channels() == 3) { 
                float valueB = 0.0f, valueG =0.0f, valueR = 0.0f;
                for (int kerCol = 0; kerCol < kerCols; ++kerCol) { 
                    int j = imCol + (kerCol - kerCenterX);
                    
                    if (j < 0 || j >= imCols) { 
                        continue;
                    } 
                    
                    float kerValue = horizontalKernel1D[kerCol];

                    cv::Vec3b imPixel = src.at<cv::Vec3b>(imRow, j);
                    valueB += imPixel[0] * kerValue;
                    valueG += imPixel[1] * kerValue;
                    valueR += imPixel[2] * kerValue;

                } 

                dst.at<cv::Vec3b>(imRow, imCol) = cv::Vec3b(valueB, valueG, valueR);
            } 
        } 
    } 
    
    return dst;
} 

cv::Mat applySeparableConvolution(const cv::Mat& src, const std::vector<float>& kernel) { 
    return applyHorizontalConvolution1D(applyVerticalConvolution1D(src, kernel), kernel);
}