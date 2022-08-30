
#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <time.h>

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

/* Compare Contour Areas */
bool compareContourAreas(vector<Point> contour1, vector<Point> contour2) {
	double i = fabs(contourArea(Mat(contour1))); // return absolute value
	double j = fabs(contourArea(Mat(contour2))); // return absolute value
	return (i < j);
}

/* Main */
int main() {
	/* Contour Detection */
	Mat crop, gray, blur, thresh;
	int minThresh = 20; // for thresholding
	int maxThresh = 255; // for thresholding
	vector<vector<Point>> contours; // for finding contours
	vector<Vec4i> hierarchy; // for finding contours
	
	/* Camera */
	/* VideoCapture cap(0); // default cam
	Mat frame; // for web
	int frameNo = 0; */

	/* Video */
	// string path = "vid/jems-stabilized.mp4"; // video path
	// string path = "vid/jun-stabilized.mp4"; // video path
	// string path = "vid/mitcham-stabilized.mp4"; // video path
	// string path = "vid/norman-stabilized.mp4"; // video path
	string path = "vid/rhys-stabilized.mp4"; // video path
	VideoCapture cap(path);
	Mat frame;
	int frameNo = 0;
	int counter = 0;
	
	/* Blink Util */
	int tempEyeState = 0;
	int fetchedClock = 0;

	while (true) {
		clock_t start = lClock();
		cap.read(frame); // read stored frame
		if (frame.empty()) break;
		
		/* Process Image */
		cvtColor(frame, gray, COLOR_BGR2GRAY); // convert to grayscale
		imshow("Grayscale", gray); // display window
		GaussianBlur(gray, blur, Size(9, 9), 0); // apply gaussian blur
		threshold(blur, thresh, minThresh, maxThresh, THRESH_BINARY_INV); // apply thresholding
		imshow("Threshold", thresh); // display window
		
		/* Contour Detection */
		findContours(thresh, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE, Point(0, 0));
		sort(contours.begin(), contours.end(), compareContourAreas);

		/* Blink Detection Accuracy Test */
		if (contours.size() == 0) {
			tempEyeState = 1;
		}
		else {
			tempEyeState = 0;
		}

		/* Print if blink */
		if (tempEyeState == 1) {
			cout << "(Close)";
			cout << " No. of Frames: " << counter++ << endl;
			fetchedClock = lClock(); // start delay
		}
		
		/* Exit at esc key */
		if (waitKey(1) == 27) {
            cout << "Program terminated." << endl;
            break;
        }
	}
	cap.release();
	destroyAllWindows();
	return 0;
}
