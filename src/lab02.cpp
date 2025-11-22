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