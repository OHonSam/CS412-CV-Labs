#include <iostream>
#include <string>
#include "Utils.hpp"
#include "HarrisDetector.hpp"
#include "BlobDetector.hpp"
#include "DoGDetector.hpp"
#include "FeatureMatching.hpp"

using namespace std;

int main(int argc, char** argv) { 
    if(argc < 2) {
        cerr << "Error: No command specified" << endl;
        displayHelp();
        return -1;
    }

    string command {argv[1]};

    if (command == "h") {
        displayHelp();
    } 
    else if (command == "harris") {
        if (argc >= 3) {
            detectHarris(argv[2]);
        } else {
            cout << "No img found. Invoke camera" << endl;
            detectHarrisCamera();
        }
    } 
    else if (command == "blob") {
        if (argc >= 3) {
            detectBlob(argv[2]);
        } else {
            cout << "No img found. Invoke camera" << endl;
            detectBlobCamera();
        }
    } 
    else if (command == "dog") {
        if (argc >= 3) {
            detectDoG(argv[2]);
        } else {
            cout << "No img found. Invoke camera" << endl;
            detectDoGCamera();
        }
    }
    else if (command == "m") {
        // Usage: ./main m <detector> <descriptor> <img1> <img2>
        if (argc >= 6) {
            matchFeatures(argv[2], argv[3], argv[4], argv[5]);
        } else if (argc == 4) {
            cout << "No images found. Invoke camera" << endl;
            matchFeaturesCamera(argv[2], argv[3]);
        } else {
            cerr << "Error: Insufficient arguments for feature matching" << endl;
            displayHelp();
            return -1;
        }
    }
    else {
        cerr << "Error: Unknown command '" << command << "'" << endl;
        displayHelp();
        return -1;
    }

    return 0;
}