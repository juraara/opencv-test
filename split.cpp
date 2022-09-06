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
	// string path = "vid/jems-stabilized.mp4"; // video path
	// string path = "vid/jun-stabilized.mp4"; // video path
	// string path = "vid/mitcham-stabilized.mp4"; // video path
	// string path = "vid/norman-stabilized.mp4"; // video path
	string path = "vid/rhys-stabilized.mp4"; // video path
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

		/* Get Height and Width */
		int width = frame.cols;
		int height = frame.rows;
	
		/* Process image */
		cvtColor(frame, gray, COLOR_BGR2GRAY); // convert to grayscale
		GaussianBlur(gray, blur, Size(9, 9), 0); // apply gaussian blur
		threshold(blur, thresh, minThresh, maxThresh, THRESH_BINARY_INV); // apply thresholding
		
		/* Get Upper and Lower Portion of Image */
		int upper_w = width;
		int upper_h = (int)((double)height * 0.60);
		int lower_w = width;
		int lower_h = height - upper_h;
		upper = thresh(Rect(0, 0, upper_w, upper_h));
		lower = thresh(Rect(0, upper_h, lower_w, lower_h));
		
		/* Display frames */
		line(gray, Point(0, upper_h), Point(width, upper_h), Scalar(0, 0, 255), 2, 8, 0);
		imshow("Grayscale", gray); // display window
		imshow("Upper", upper);
		imshow("Lower", lower);
		
		/* Calculate Histogram */
		MatND upperHistogram, lowerHistogram;
		int histSize = 256;
		const int* channelNumbers = { 0 };
		float channelRange[] = { 0.0, 256.0 };
		const float* channelRanges = channelRange;
		int numberBins = 256;
		
		/* Histogram for Upper Eye */
		calcHist(&upper, 1, 0, Mat(), upperHistogram, 1, &numberBins, &channelRanges);
		/* Histogram for Lower Eye */
		calcHist(&lower, 1, 0, Mat(), lowerHistogram, 1, &numberBins, &channelRanges);
		
		/* Compare Histograms */
		float lowerPixels = lowerHistogram.at<float>(255);
		float upperPixels = upperHistogram.at<float>(255);

		/* Print if blink */
		if (upperPixels < lowerPixels) {
			cout << "(Close)";
			cout << " Counter: " << counter++ << endl;
		}
		
		/* Exit at esc key */
		if (waitKey(1) == 27) {
            cout << "Program terminated." << endl;
            break;
        }
	}
	
	return 0;
}
