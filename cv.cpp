/* Image Processing Module
 * (Contour Detection) 
 * Note: If the pixel value is smaller than the threshold, it is set to 0, otherwise it is set to a maximum value */
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
	/* PERCLOS */
	int eyeState[65]; // 65 frames in 14fps should make 5fps for PERCLOS
	double perclos = 0; // store result
	int counter;
	// double counterTime = 0; // 5s counter
	// int overallTime = 0; // overall time
	
	/* Contour Detection */
	Mat crop, gray, blur, thresh;
	int minThresh = 20; // for thresholding
	int maxThresh = 255; // for thresholding
	vector<vector<Point>> contours; // for finding contours
	vector<Vec4i> hierarchy; // for finding contours
	
	/* Camera */
	VideoCapture cap(0); // default cam
	Mat frame; // store frames here
	int frameNo = 0;
	
	
	/* Blink Util */
	int tempEyeState = 0;

	while (true) {
		clock_t start = lClock();
		
		cap.read(frame); // read stored frame
		// imshow("Image", frame); // window
		if (frame.empty()) break;
		
		/* Calculate Histogram */
		crop = frame(Rect(170, 180, 230, 140)); // crop frame
		// imshow("Crop", crop); // display window
		cvtColor(crop, gray, COLOR_BGR2GRAY); // convert to grayscale
		imshow("Grayscale", gray); // display window
		GaussianBlur(gray, blur, Size(9, 9), 0); // apply gaussian blur
		// imshow("GaussianBlur", blur); // display window
		threshold(blur, thresh, minThresh, maxThresh, THRESH_BINARY_INV); // apply thresholding
		imshow("Threshold", thresh); // display window
		
		findContours(thresh, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE, Point(0, 0));
		sort(contours.begin(), contours.end(), compareContourAreas);
		
		/* Create delay (500ms) after blink is detected (testing) */
		/* if ((lClock() - fetchedClock) > 500) {
			fetchedClock = lClock(); // start delay
		} */
		
		if (contours.size() == 0) {
			tempEyeState = 1;
			eyeState[counter] = 1;  // blink
		}
		else {
			tempEyeState = 0;
			eyeState[counter] = 0; // blink~janai
		}
		
		counter++; // increment counter
		if(counter == 65) {
			counter = 0;
			/* Calculate PERCLOS */
			int sum = 0;
			for (int j = 0; j < 65; j++) {
				sum += eyeState[j];
			}
			perclos = (sum/65.0) * 100;
		}
		
		/* Print to Terminal */
		double duration = lClock() - start; // stop counting
		double averageTimePerFrame = averageDuration(duration); // avg time per frame
		if (tempEyeState == 1) { 
			cout << "(Close)" << " Avg tpf: " << averageTimePerFrame << "ms" << " Avg fps: " << averageFps() << " Perclos: " << perclos << " Frame no: " << frameNo++ << endl; 
		} else {
			cout << "(Open)" << " Avg tpf: " << averageTimePerFrame << "ms" << " Avg fps: " << averageFps() << " Perclos: " << perclos << " Frame no: " << frameNo++ << endl;
		}
		
		/* Exit at esc key */
		if (waitKey(1) == 27) {
            cout << "Program terminated." << endl;
            break;
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
