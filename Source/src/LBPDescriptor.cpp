#include "LBPDescriptor.hpp"
#include <opencv2/opencv.hpp>

void createLBPTrackbars(const std::string& windowName, LBPContext* ctx, void (*onCallback)(int, void*)) {
    cv::createTrackbar("LBP Radius", windowName, &ctx->radius, 10, onCallback, ctx);
    cv::createTrackbar("LBP Patch Size", windowName, &ctx->patchSize, 64, onCallback, ctx);
}

void computeLBPDescriptors(const cv::Mat& gray, 
                            const std::vector<cv::KeyPoint>& keypoints, 
                            cv::Mat& descriptors,
                            const LBPContext& context
                        ) {
    const int radius = context.radius;
    const int patchSize = context.patchSize;
    descriptors = cv::Mat::zeros(static_cast<int>(keypoints.size()), 256, CV_32F);

    for (size_t i = 0; i < keypoints.size(); ++i) {
        const cv::KeyPoint& kp = keypoints[i];
        int cx = static_cast<int>(kp.pt.x);
        int cy = static_cast<int>(kp.pt.y);
        
        int x1 = std::max(radius, cx - patchSize / 2);
        int y1 = std::max(radius, cy - patchSize / 2);
        int x2 = std::min(gray.cols - radius - 1, cx + patchSize / 2);
        int y2 = std::min(gray.rows - radius - 1, cy + patchSize / 2);

        // Skip if patch is invalid
        if (x1 >= x2 || y1 >= y2) {
            descriptors.row(i).setTo(1.0f / 256.0f); // Uniform distribution
            continue;
        }

        for (int y = y1; y < y2; ++y) {
            for (int x = x1; x < x2; ++x) {
                uchar center = gray.at<uchar>(y, x);
                uchar lbpCode = 0;

                lbpCode |= (gray.at<uchar>(y - 1, x - 1) >= center) << 7;
                lbpCode |= (gray.at<uchar>(y - 1, x) >= center) << 6;
                lbpCode |= (gray.at<uchar>(y - 1, x + 1) >= center) << 5;
                lbpCode |= (gray.at<uchar>(y, x + 1) >= center) << 4;
                lbpCode |= (gray.at<uchar>(y + 1, x + 1) >= center) << 3;
                lbpCode |= (gray.at<uchar>(y + 1, x) >= center) << 2;
                lbpCode |= (gray.at<uchar>(y + 1, x - 1) >= center) << 1;
                lbpCode |= (gray.at<uchar>(y, x - 1) >= center) << 0;

                descriptors.at<float>(static_cast<int>(i), lbpCode) += 1.0f;
            }
        }

        // Safe normalization
        double sum = cv::sum(descriptors.row(i))[0];
        if (sum > 1e-6) {
            cv::normalize(descriptors.row(i), descriptors.row(i), 1.0, 0.0, cv::NORM_L1);
        } else {
            descriptors.row(i).setTo(1.0f / 256.0f);
        }
    }
}