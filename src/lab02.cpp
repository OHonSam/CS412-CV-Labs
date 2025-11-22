#include <iostream>
#include <string>
#include "FeatureUtils.hpp"

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
    else if (command == "harris" && argc >= 3) {
        detectHarris(argv[2]);
    } 
    else if (command == "blob" && argc >= 3) {
        detectBlob(argv[2]);
    } 
    else if (command == "dog" && argc >= 3) {
        detectDoG(argv[2]);
    }
    else if (command == "m" && argc >= 6) {
        // Usage: ./main m <detector> <descriptor> <img1> <img2>
        matchFeatures(argv[2], argv[3], argv[4], argv[5]);
    }
    else {
        cout << "No img found. Invoke camera" << endl;
        openCamera();
    }

    return 0;
}