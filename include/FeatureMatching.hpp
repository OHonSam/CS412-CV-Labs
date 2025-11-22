#pragma once
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

// Matching functions
void matchFeatures(const std::string& detectorType, const std::string& descriptorType, 
                  const std::string& img1Path, const std::string& img2Path);