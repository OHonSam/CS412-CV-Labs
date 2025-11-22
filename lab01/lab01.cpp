#include "iostream" 
#include "opencv2/highgui/highgui.hpp" 
#include "opencv2/opencv.hpp" 
#include "string" 

using namespace std;
using namespace cv;

Mat image;
bool toGray = false;
bool addBrightnessTrack = false;
bool addContrastTrack = false;
bool avgFilter = false;
bool gaussFilter = false;

void MyCallbackForBrightness(int iValueForBrightness, void* userData); 
void MyCallbackForContrast(int iValueForContrast, void* userData); 
void MyCallbackForAvgFilter(int iKernelSize, void* userData); 
void MyCallbackForGaussFilter(int iKernelSize, void* userData); 
Mat createAverageFilter(int size); 
Mat createGaussianFilter(int size); 
Mat applyConvolution(const Mat& src, const Mat& kernel); 
vector<float> createAverageFilter1D(int size); 
vector<float> createGaussianFilter1D(int size); 
Mat applyHorizontalConvolution1D(const Mat& src, const vector<float>& horizontalKernel1D);
Mat applyVerticalConvolution1D(const Mat& src, const vector<float>& verticalKernel1D);
Mat applySeparableConvolution(const Mat& src, const vector<float>& kernel);


int main(int argc, char** argv) { 
    if (argc < 3) { 
        cout << "Not enough input arguments! Please input following the format " 
                "<Program.exe> -rgb2gray <FileNameInput>" << endl;
        return 1;
    } 
    
    for (int i = 1; i < argc - 1; ++i) { 
        string flag = argv[i];
        if (flag == "-rgb2gray") { toGray = true; } 
        else if (flag == "-brightness") { addBrightnessTrack = true; } 
        else if (flag == "-contrast") { addContrastTrack = true; } 
        else if (flag == "-avg") { avgFilter = true; } 
        else if (flag == "-gauss") { gaussFilter = true; } 
    } 

    string fileNameInput{argv[argc - 1]};
    Mat original_image = imread(fileNameInput);
    image = imread(fileNameInput, toGray ? cv::IMREAD_GRAYSCALE : cv::IMREAD_COLOR);

    if (image.empty()) { cout << "Error: Could not load image " << fileNameInput << endl;
        return 1;
    } 

    namedWindow("Output Image");
    imshow("Output Image", image);

    int iValueForBrightness = 50;
    int iValueForContrast = 50;
    int iKernelSize = 3;

    if (addBrightnessTrack) { 
        createTrackbar( "Brightness", "Output Image", &iValueForBrightness, 100, MyCallbackForBrightness, &iValueForContrast );
    } 
    if (addContrastTrack) { 
        createTrackbar( "Contrast", "Output Image", &iValueForContrast, 100, MyCallbackForContrast, &iValueForBrightness );
    } 
    if (avgFilter) { 
        createTrackbar( "Average Filter", "Output Image", &iKernelSize, 30, MyCallbackForAvgFilter, nullptr );
    } 
    if (gaussFilter) { 
        createTrackbar( "Gaussian Filter", "Output Image", &iKernelSize, 30, MyCallbackForGaussFilter, nullptr );
    } 
    
    imshow("Input Image", original_image);
    waitKey(0);

    return 0;
} // Compile: g++ -std=c++17 -o lab01.exe lab01.cpp pkg-config --cflags --libs opencv4 

void MyCallbackForBrightness(int iValueForBrightness, void* userData) { 
    int iValueForContrast = *(static_cast<int*>(userData));

    // Calculating brightness and contrast value 
    int iBrightness = iValueForBrightness - 50;
    double dContrast = iValueForContrast / 50.0;

    // Calculated contrast and brightness value 
    cout << "MyCallbackForBrightness : Contrast=" << dContrast << ", Brightness=" << iBrightness << endl;

    // adjust the brightness and contrast 
    // Mat result
    // image.convertTo(result, -1, dContrast, iBrightness);

    Mat result = Mat(image.rows, image.cols, image.type(), Scalar(0));

    int imRows = image.rows;
    int imCols = image.cols;
    int imChannels = image.channels();

    for (int y = 0; y < imRows; ++y) {
        for (int x = 0; x < imCols; ++x) {
            for (int c = 0; c < imChannels; ++c) {
                result.at<Vec3b>(y, x)[c] = saturate_cast<uchar>(dContrast * image.at<Vec3b>(y, x)[c] + iBrightness);
            }
        }
    }

    // show the brightness and contrast adjusted image 
    imshow("Output Image", result);
} 

void MyCallbackForContrast(int iValueForContrast, void* userData) { 
    int iValueForBrightness = *(static_cast<int*>(userData));

    // Calculating brightness and contrast value 
    int iBrightness = iValueForBrightness - 50;
    double dContrast = iValueForContrast / 50.0;

    // Calculated contrast and brightness value 
    cout << "MyCallbackForContrast : Contrast=" << dContrast << ", Brightness=" << iBrightness << endl;

    // adjust the brightness and contrast 
    // Mat result;
    // image.convertTo(result, -1, dContrast, iBrightness);

    Mat result = Mat(image.rows, image.cols, image.type(), Scalar(0));

    int imRows = image.rows;
    int imCols = image.cols;
    int imChannels = image.channels();

    for (int y = 0; y < imRows; ++y) {
        for (int x = 0; x < imCols; ++x) {
            for (int c = 0; c < imChannels; ++c) {
                result.at<Vec3b>(y, x)[c] = saturate_cast<uchar>(dContrast * image.at<Vec3b>(y, x)[c] + iBrightness);
            }
        }
    }

    // show the brightness and contrast adjusted image 
    imshow("Output Image", result);
} 

void MyCallbackForAvgFilter(int iKernelSize, void* userData) { 
    // Ensure kernel size is at least 3 and odd 
    if (iKernelSize % 2 == 0) { 
        iKernelSize += 1;
    } 
    // Use OpenCV's blur function for average filtering 
    // blur(image, result, Size(iKernelSize, iKernelSize));

    // Mat avgKernel = createAverageFilter(iKernelSize);
    // Mat result = applyConvolution(image, avgKernel);

    vector<float> avgKernel = createAverageFilter1D(iKernelSize);
    Mat result = applySeparableConvolution(image, avgKernel);
    cout << "Average Filter: Kernel Size=" << iKernelSize << endl;

    // Show the filtered image 
    imshow("Output Image", result);
} 

void MyCallbackForGaussFilter(int iKernelSize, void* userData) { 
    // Ensure kernel size is at least 3 and odd 
    if (iKernelSize % 2 == 0) { 
        iKernelSize += 1;
    } 
    // Use OpenCV's GaussianBlur function 
    // Sigma is calculated automatically based on kernel size 
    // GaussianBlur(image, result, Size(iKernelSize, iKernelSize), 0);

    // Mat gaussianKernel = createGaussianFilter(iKernelSize);
    // Mat result = applyConvolution(image, gaussianKernel);

    vector<float> gaussianKernel = createGaussianFilter1D(iKernelSize);
    Mat result = applySeparableConvolution(image, gaussianKernel);
    cout << "Gaussian Filter: Kernel Size=" << iKernelSize << endl;

    // Show the filtered image 
    imshow("Output Image", result);
} 

Mat createAverageFilter(int size) { 
    return Mat(size, size, CV_32F, 1.0f / (size * size));
} 

Mat createGaussianFilter(int size) { 
    // https://docs.opencv.org/2.4/modules/imgproc/doc/filtering.html#getgaussiankernel 
    float sigma = 0.3 * ((size - 1) * 0.5 -1) + 0.8;
    float alpha = 0.0f;
    int centerIdx = (size - 1) / 2;

    Mat kernel = Mat(size, size, CV_32F, Scalar(1));

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

Mat applyConvolution(const Mat& src, const Mat& kernel) { 
    int imRows = src.rows;
    int imCols = src.cols;
    
    int kerRows = kernel.rows;
    int kerCols = kernel.cols;
    int kerCenterY = (kerRows - 1)/2;
    int kerCenterX = (kerCols - 1)/2;

    Mat dst = Mat(imRows, imCols, src.type(), Scalar(1));
    
    for (int imRow = 0; imRow < imRows; ++imRow) { 
        for (int imCol = 0; imCol < imCols; ++imCol) { 
            // Use zero paddings so [col][row] will be at the center of kernel 
            if (src.channels() == 1) { 
                float value = 0.0f;

                for (int kerRow = 0; kerRow < kerRows; +kerRow) { 
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
                dst.at<uchar>(imRow, imCol) = saturate_cast<uchar>(value);
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

                        Vec3b imPixel = src.at<Vec3b>(i, j);
                        valueB += imPixel[0] * kerValue;
                        valueG += imPixel[1] * kerValue;
                        valueR += imPixel[2] * kerValue;
                    } 
                } 
                
                // dst.at<Vec3b>(imRow, imCol) =Vec3b(valueB, valueG, valueR);

                // Clamp to valid range [0, 255] 
                dst.at<Vec3b>(imRow, imCol) = Vec3b( 
                    saturate_cast<uchar>(valueB), 
                    saturate_cast<uchar>(valueG), 
                    saturate_cast<uchar>(valueR) 
                );
            } 
        } 
    } 
    
    return dst;
} 

vector<float> createAverageFilter1D(int size) { 
    return vector<float>(size, 1.0f / size);
} 

vector<float> createGaussianFilter1D(int size) { 
    float sigma = 0.3 * ((size - 1) * 0.5 -1) + 0.8;
    float alpha = 0.0f;
    int centerIdx = (size - 1) / 2;

    vector<float> kernel(size);

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

Mat applyVerticalConvolution1D(const Mat& src, const vector<float>& verticalKernel1D) { 
    int imRows = src.rows;
    int imCols = src.cols;

    int kerRows = verticalKernel1D.size();
    int kerCenterY = (kerRows - 1)/2;

    Mat dst = Mat(imRows, imCols, src.type(), Scalar(1));

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

                    Vec3b imPixel = src.at<Vec3b>(i, imCol);
                    valueB += imPixel[0] * kerValue;
                    valueG += imPixel[1] * kerValue;
                    valueR += imPixel[2] * kerValue;

                } 
                
                dst.at<Vec3b>(imRow, imCol) =Vec3b(valueB, valueG, valueR);
            } 
        } 
    } 
    return dst;
} 

Mat applyHorizontalConvolution1D(const Mat& src, const vector<float>& horizontalKernel1D) { 
    int imRows = src.rows;
    int imCols = src.cols;
    int kerCols = horizontalKernel1D.size();
    int kerCenterX = (kerCols - 1)/2;

    Mat dst = Mat(imRows, imCols, src.type(), Scalar(1));

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

                    Vec3b imPixel = src.at<Vec3b>(imRow, j);
                    valueB += imPixel[0] * kerValue;
                    valueG += imPixel[1] * kerValue;
                    valueR += imPixel[2] * kerValue;

                } 
                
                dst.at<Vec3b>(imRow, imCol) =Vec3b(valueB, valueG, valueR);
            } 
        } 
    } 
    
    return dst;
} 

Mat applySeparableConvolution(const Mat& src, const vector<float>& kernel) { 
    return applyHorizontalConvolution1D(applyVerticalConvolution1D(src, kernel), kernel);
}