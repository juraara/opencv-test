/* Image Processing Module
 * (Global Thresholding of Negative) 
 * 1. Good for blink detection
 * 2. When determining open and closed states, make sure to place camera abover lower eyelid. Lower eyelid may produce significant number of black pixels greater or equal than the number of black pixels the pupil+iris may produce.
 * 3. You can also lower the minimum threshold to fix the problem (should have comparable accuracy with contour detection)*/

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
	/* Camera */
	// VideoCapture cap(0); // default cam
	Mat crop, gray, blur, thresh;
	Mat upper; // (Custom)
	int minThresh = 70; // for thresholding
	int maxThresh = 255; // for thresholding

	/* Video */
	// string path = "vid/jems-stabilized.mp4"; // video path
	// string path = "vid/jun-stabilized.mp4"; // video path
	// string path = "vid/mitcham-stabilized.mp4"; // video path
	string path = "vid/norman-stabilized.mp4"; // video path
	// string path = "vid/rhys-stabilized.mp4"; // video path
	VideoCapture cap(path);
	Mat frame;
	int frameNo = 0;
	int counter = 0;
	int tempEyeState = 0;
	
	/* Black pixel values */
	float prevFrame = 0.0;
	float currentFrame = 0.0;
	float percentageDifference = 0.0;
	
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

		/* Get Upper Portion of Image */
		int upper_w = width;
		int upper_h = (int)((double)height * 0.60);
		upper = thresh(Rect(0, 0, upper_w, upper_h));
		
		/* Display frames */
		line(gray, Point(0, upper_h), Point(upper_w, upper_h), Scalar(0, 0, 255), 2, 8, 0);
		imshow("Grayscale", gray); // display window
		imshow("Upper", upper);
		
		/* Calculate Histogram */
		MatND histogram;
		int histSize = 256;
		const int* channelNumbers = { 0 };
		float channelRange[] = { 0.0, 256.0 };
		const float* channelRanges = channelRange;
		int numberBins = 256;
		
		int histWidth = 512;
		int histHeight = 400;
		int binWidth = cvRound((double)histWidth / histSize);
		
		calcHist(&upper, 1, 0, Mat(), histogram, 1, &numberBins, &channelRanges);
		Mat histImage(histHeight, histWidth, CV_8UC3, Scalar(0, 0, 0));
		normalize(histogram, histogram, 0, histImage.rows, NORM_MINMAX, -1, Mat());
		
		/* Compare Histogram Value from Previous Frame */
		prevFrame = currentFrame;
		currentFrame = histogram.at<float>(255);
		percentageDifference = ((prevFrame - currentFrame) / ((prevFrame + currentFrame) / 2)) * 100;

		/* Check Percentage Difference */
		if (percentageDifference >= 80.0) {
			tempEyeState = 1;
		} else if (percentageDifference <= -80.0) {
			tempEyeState = 0;
		}

		/* Print if blink */
		if (tempEyeState == 1) {
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
