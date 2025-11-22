#include "Utils.hpp" 
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