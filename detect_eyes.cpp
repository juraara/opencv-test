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
	/* Camera */
	VideoCapture cap(0); // default cam
	Mat frame, crop, gray; // store frames here
	int frameNo = 0;
	
	/* Eye Detection */
	CascadeClassifier faceCascade;
	vector<Rect> faces;
	faceCascade.load("./haarcascade_eye_tree_eyeglasses.xml");
    int faceDetectDelay = 0;


	while (true) {
		cap.read(frame); // read stored frame
		if (frame.empty()) break;
		// crop = frame(Rect(0, 0, 480, 480)); // crop frame
		cvtColor(frame, gray, COLOR_BGR2GRAY); // convert to grayscale

		if (faceDetectDelay < 30) {
			faceDetectDelay++;
		} else {
			faceDetectDelay = 0;
			vector<Rect> faces;
			faceCascade.detectMultiScale(gray, faces, 1.1, 10);
			for (int i = 0; i < faces.size(); i++) {
				rectangle(gray, faces[i].tl(), faces[i].br(), Scalar(255, 255, 255), 2);
			}
		}

		/* faceDetectDelay = 0;
		vector<Rect> faces;
		faceCascade.detectMultiScale(gray, faces, 1.1, 10);
		for (int i = 0; i < faces.size(); i++) {
			rectangle(gray, faces[i].tl(), faces[i].br(), Scalar(255, 255, 255), 2);
		} */
		line(frame, Point(320, 0), Point(320, 480), Scalar(0, 255, 0), 1, 8, 0);
		line(frame, Point(0, 240), Point(640, 240), Scalar(0, 255, 0), 1, 8, 0);
		imshow("Grayscale", frame); // display
		// cout << "Width : " << gray.cols << endl;
		// cout << "Height: " << gray.rows << endl;

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

