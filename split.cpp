/* Image Processing Module
 * (Eye Black Pixel Ratio Analysis) 
 * 1. Performs well */

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <vector>

using namespace cv;
using namespace std;

double _averageDuration = 0; // average duration (averageDuration)
int _fetchedClock = 0; // gets lClock() at 1s (averageFps)
double _averageFps = 0; // average fps (averageFps)
double _frameNo = 0; // no of frames in 1s (averageFps)

/* Clock */
int lClock() {
	struct timespec ts;
	clock_gettime(CLOCK_MONOTONIC, &ts);
	// cout << "lClock: " << (ts.tv_sec * 1000) + (ts.tv_nsec * 1e-6) << endl; 
	return (ts.tv_sec * 1000) + (ts.tv_nsec * 1e-6);
}

/* Average Duration */
double averageDuration(double newDuration) {
	_averageDuration = 0.98 * _averageDuration + 0.02 * newDuration;
	return _averageDuration;
}	

/* Average fps */
double averageFps() {
	if ((lClock() - _fetchedClock) > 1000) {
		_fetchedClock = lClock();
		_averageFps = 0.7 * _averageFps + 0.3 * _frameNo;
		// cout << "fps: " << _frameNo << endl;
		_frameNo = 0;
	}
	_frameNo++;
	return _averageFps;
}

int main() {
	/* PERCLOS */
	int eyeState[65]; // 65 frames in 14fps should make 5fps for PERCLOS
	double perclos = 0; // store result
	int counter;
	
	/* Camera */
	// VideoCapture cap(0); // default cam
	Mat crop, gray, blur, thresh;
	Mat upper, lower; // upper and lower section of eyes
	int minThresh = 70; // for thresholding
	int maxThresh = 255; // for thresholding
	int frameNo = 0;

	/* Video */
	string path = "vid/norman-test.mp4";
	VideoCapture cap(path);
	Mat frame;

	/* Print Utils */
	int tempEyeState = 0;

	/* Blink Util */
	int fetchedClock = 0;
	
	while(true) {
		clock_t start = lClock(); // start counting
		cap.read(frame);
		if (frame.empty()) break;
	
		/* Process image */
		// crop = frame(Rect(170, 180, 230, 140)); // crop frame
		// line(frame, Point(0, 70), Point(228, 70), Scalar(0, 0, 255), 2, 8, 0);
		cvtColor(frame, gray, COLOR_BGR2GRAY); // convert to grayscale
		GaussianBlur(gray, blur, Size(9, 9), 0); // apply gaussian blur
		threshold(blur, thresh, minThresh, maxThresh, THRESH_BINARY_INV); // apply thresholding
		
		int upper_length = 90;
		int lower_length = 140 - upper_length;
		upper = thresh(Rect(0, 0, 228, upper_length));
		imshow("Upper", upper);
		lower = thresh(Rect(0, upper_length, 228, lower_length));
		imshow("Lower", lower);
		
		/* Display frames */
		line(gray, Point(0, upper_length), Point(228, upper_length), Scalar(0, 0, 255), 2, 8, 0);
		imshow("Grayscale", gray); // display window
		imshow("Upper", upper);
		
		/* Calculate Histogram */
		MatND upperHistogram, lowerHistogram;
		int histSize = 256;
		const int* channelNumbers = { 0 };
		float channelRange[] = { 0.0, 256.0 };
		const float* channelRanges = channelRange;
		int numberBins = 256;
		
		int histWidth = 512;
		int histHeight = 400;
		int binWidth = cvRound((double)histWidth / histSize);
		
		/* Histogram for Upper Eye */
		calcHist(&upper, 1, 0, Mat(), upperHistogram, 1, &numberBins, &channelRanges);
		/* Histogram for Lower Eye */
		calcHist(&lower, 1, 0, Mat(), lowerHistogram, 1, &numberBins, &channelRanges);
		
		/* Compare Histograms */
		float lowerPixels = lowerHistogram.at<float>(255);
		float upperPixels = upperHistogram.at<float>(255);
		
		// if (upperPixels < lowerPixels) {
		// 	eyeState[counter] = 1;
		// 	tempEyeState = 1;
		// } else {
		// 	eyeState[counter] = 0;
		// 	tempEyeState = 0;
		// }
		
		// counter++; // increment counter
		// if(counter == 65) {
		// 	counter = 0;
		// 	/* Calculate PERCLOS */
		// 	int sum = 0;
		// 	for (int j = 0; j < 65; j++) {
		// 		sum += eyeState[j];
		// 	}
		// 	perclos = (sum/65.0) * 100;
		// }
		
		// /* Print to Terminal */
		// double duration = lClock() - start; // stop counting
		// double averageTimePerFrame = averageDuration(duration); // avg time per frame
		// if (tempEyeState == 1) { 
		// 	cout << "(Close)" << " Avg tpf: " << averageTimePerFrame << "ms" << " Avg fps: " << averageFps() << " Perclos: " << perclos << " Frame no: " << frameNo++ << endl; 
		// } else {
		// 	cout << "(Open)" << " Avg tpf: " << averageTimePerFrame << "ms" << " Avg fps: " << averageFps() << " Perclos: " << perclos << " Frame no: " << frameNo++ << endl;
		// }

		/* Create delay (200ms) after blink is detected (testing) */
		if ((lClock() - fetchedClock) > 500) {
			if (upperPixels < lowerPixels) {
				tempEyeState = 1;
				cout << "(Close)";
				cout << " Counter: " << counter++ << endl;
				fetchedClock = lClock(); // start delay
			} else {
				tempEyeState = 0;
			}
		}
		
		/* Exit at esc key */
		if (waitKey(1) == 27) {
            cout << "Program terminated." << endl;
            break;
        }
	}
	
	return 0;
}
