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
	/* PERCLOS */
	int eyeState[65]; // 65 frames in 14fps should make 5fps for PERCLOS
	double perclos = 0; // store result
	int counter = 0;
	
	/* Camera */
	VideoCapture cap(0); // default cam
	Mat frame, crop, gray, blur, thresh;
	Mat upper; // (Custom)
	int minThresh = 70; // for thresholding
	int maxThresh = 255; // for thresholding
	int frameNo = 0;
	
	/* Black pixel values */
	float prevFrame = 0.0, currentFrame = 0.0, percentageDifference = 0.0;
	
	/* Print Util */
	int tempEyeState = 0;
	
	while(true) {
		clock_t start = lClock(); // start counting
		cap.read(frame);
		// imshow("Frame", frame);
	
		/* Process image */
		crop = frame(Rect(170, 180, 230, 140)); // crop frame
		// imshow("Crop", crop); // display window
		cvtColor(crop, gray, COLOR_BGR2GRAY); // convert to grayscale
		GaussianBlur(gray, blur, Size(9, 9), 0); // apply gaussian blur
		// imshow("GaussianBlur", blur); // display window
		threshold(blur, thresh, minThresh, maxThresh, THRESH_BINARY_INV); // apply thresholding
		// imshow("Threshold", thresh); // display window
		upper = thresh(Rect(0, 0, 230, 70));
		
		/* Display frames */
		line(gray, Point(0, 70), Point(230, 70), Scalar(0, 0, 255), 2, 8, 0);
		imshow("Grayscale", gray); // display window
		imshow("Upper", upper);
		
		/* Calculate Histogram */
		MatND histogram;
		int histSize = 256;
		const int* channelNumbers = { 0 };
		float channelRange[] = { 0.0, 256.0 };
		const float* channelRanges = channelRange;
		int numberBins = 256;
		
		// set histogram size
		int histWidth = 512, histHeight = 400;
		int binWidth = cvRound((double)histWidth / histSize);
		
		calcHist(&upper, 1, 0, Mat(), histogram, 1, &numberBins, &channelRanges);
		Mat histImage(histHeight, histWidth, CV_8UC3, Scalar(0, 0, 0));
		normalize(histogram, histogram, 0, histImage.rows, NORM_MINMAX, -1, Mat());
		
		/* Create Histogram Figure */
		/* for (int i = 1; i < histSize; i++) {
			line(histImage, Point(binWidth * (i - 1), histHeight = cvRound(histogram.at<float>(i - 1))),
			Point(binWidth * (i), histHeight - cvRound(histogram.at<float>(i))),
			Scalar(255, 0, 0), 2, 8, 0);
		}
		imshow("Histogram", histImage); */
		
		/* Compare Histogram Value from Previous Frame */
		prevFrame = currentFrame;
		currentFrame = histogram.at<float>(255);
		percentageDifference = ((prevFrame - currentFrame) / ((prevFrame + currentFrame) / 2)) * 100;
		
		/* Switch Eye State Based on Percentage Difference */
		if (percentageDifference >= 80.0) {
			tempEyeState = 1;
		} else if (percentageDifference <= -80.0) {
			tempEyeState = 0;
		}	
		
		/* Record Eye State */
		if (tempEyeState == 1) { 
			eyeState[counter] = 1;
		} else {
			eyeState[counter] = 0;
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
	}
	
	return 0;
}
