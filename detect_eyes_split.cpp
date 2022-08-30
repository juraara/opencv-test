#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect/objdetect.hpp>
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
	/* Contour Detection */
	Mat crop, gray, blur, thresh;
	int minThresh = 70; // for thresholding
	int maxThresh = 255; // for thresholding
	
	/* Camera */
	VideoCapture cap(0); // default cam
	Mat frame; // store frames here
	int frameNo = 0;
	
	/* Eye Detection */
	CascadeClassifier faceCascade;
	vector<Rect> faces;
	Mat eyeROI;
	Mat croppedEye;
	faceCascade.load("./haarcascade/haarcascade_eye_tree_eyeglasses.xml");
	int height = 0, width = 0;

	float prevFrame = 0.0;
	float currentFrame = 0.0;
	float percentageDifference = 0.0;
	int delay = 0; // delay

	/* Histogram Utils */
	MatND currentHist, prevHist;
	int histSize = 256;
	const int* channelNumbers = { 0 };
	float channelRange[] = { 0.0, 256.0 };
	const float* channelRanges = channelRange;
	int numberBins = 256;

	/* Init */
	cap.read(frame); // read stored frame
	crop = frame(Rect(170, 180, 230, 140)); // crop frame
	cvtColor(crop, gray, COLOR_BGR2GRAY); // convert to grayscale
	calcHist(&gray, 1, 0, Mat(), currentHist, 1, &numberBins, &channelRanges);
	calcHist(&gray, 1, 0, Mat(), prevHist, 1, &numberBins, &channelRanges);
	

	while (true) {
		cap.read(frame); // read stored frame
		if (frame.empty()) break;
		
		crop = frame(Rect(170, 180, 230, 140)); // crop frame
		cvtColor(crop, gray, COLOR_BGR2GRAY); // convert to grayscale

		/* Calculate Histogram */
		if (delay < 15) {
			delay++;
			calcHist(&gray, 1, 0, Mat(), currentHist, 1, &numberBins, &channelRanges);
		} else {
			delay = 0;
			double histMatchingCorrelation = compareHist(currentHist, prevHist, HISTCMP_CORREL);
			cout << "Matching Correlation: " << histMatchingCorrelation << endl;
			
			/* Detect Eyes if Hist Value Changes */
			if (histMatchingCorrelation < 0.95) {
				vector<Rect> faces;
				faceCascade.detectMultiScale(gray, faces, 1.1, 10);
				for (int i = 0; i < faces.size(); i++) {
					eyeROI = gray(faces[i]);
					height = faces[i].height; cout << "Height: " << height << endl;
					width = faces[i].width; cout << "Widht: " << width << endl;
					croppedEye = eyeROI;
					rectangle(gray, faces[i].tl(), faces[i].br(), Scalar(255, 255, 255), 2);
				}
			}
			calcHist(&gray, 1, 0, Mat(), prevHist, 1, &numberBins, &channelRanges);
		}
		imshow("Graysclade", gray); // display window
		if (!croppedEye.empty()) {
			imshow("Cropped Eye", croppedEye); // display window
			// Mat histEqualized;
			// equalizeHist(croppedEye, histEqualized); // equalize
			GaussianBlur(croppedEye, blur, Size(9, 9), 0); // apply gaussian blur
			threshold(blur, thresh, minThresh, maxThresh, THRESH_BINARY_INV); 

			int lowerHeight = (int)((double)height*0.5);
			int upperHeight = (int)((double)height*0.5);
			int upperY = (int)((double)height*0.2);
			int lowerY = upperHeight;
			Mat upper = thresh(Rect(0, upperY, width, upperHeight));
			Mat lower = thresh(Rect(0, lowerY, width, lowerHeight));

			/* Calculate Histogram for Upper and Lower Eye */
			MatND upperHist, lowerHist;
			calcHist(&upper, 1, 0, Mat(), upperHist, 1, &numberBins, &channelRanges);
			calcHist(&lower, 1, 0, Mat(), lowerHist, 1, &numberBins, &channelRanges);
			float lowerPixels = lowerHist.at<float>(255);
			float upperPixels = upperHist.at<float>(255);

			if (upperPixels < lowerPixels) {
				cout << "close" << "fps" << averageFps() << endl;
			} else {
				cout << "open" << "fps" << averageFps() << endl;
			}
			imshow("Upper", upper);
			imshow("Lower", lower);
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

