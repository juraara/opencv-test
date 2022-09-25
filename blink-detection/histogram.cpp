
#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <time.h>

using namespace cv;
using namespace std;

/* Clock */
int getClock() {
	struct timespec ts;
	clock_gettime(CLOCK_MONOTONIC, &ts);
	return (ts.tv_sec * 1000) + (ts.tv_nsec * 1e-6);
}

void gammaCorrection(const Mat &src, Mat &dst, const float gamma) {
	Mat gray;
    cvtColor(src, gray, COLOR_BGR2GRAY);

	// double mid = 0.5;
    // cv::Scalar temp = cv::mean(hsv_pl-anes[2]);
    // float mean = temp.val[0];
    // double gamma = log(mid*255)/log(mean);

    float invGamma = 1 / gamma;
    Mat table(1, 256, CV_8U);
    uchar *p = table.ptr();
    for (int i = 0; i < 256; ++i) {
        p[i] = (uchar) (pow(i / 255.0, invGamma) * 255);
    }
	dst = src.clone();
    LUT(src, table, dst);
}

std::vector<cv::Rect> eyes;

void detectEyes(Mat &frame, CascadeClassifier &eyeCascade) {
	Mat gray;
	cvtColor(frame, gray, COLOR_BGR2GRAY); // convert image to grayscale
	equalizeHist(gray, gray); // enchance image contrast

    eyeCascade.detectMultiScale(gray, eyes, 1.1, 2, 0 | CASCADE_SCALE_IMAGE, cv::Size(30, 30));
	if (eyes.size() == 0) return; // no eye was detected
	for (cv::Rect &eye : eyes) {
        rectangle(frame, eye.tl(), eye.br(), cv::Scalar(0, 255, 0), 2);
    }
	Mat eye = gray(eyes[0]); // crop the eye
	imshow("eye", eye);
}

int frameNo = 0;
int blinkWindow = 0;

int detectedBlinks[50];
int blinkCounter = 0;

float prevFrameWhitePixelNo = 0.0;
float currFrameWhitePixelNo = 0.0;
bool close = false;

void detectBlink(Mat &frame) {
	Mat gray;
	cvtColor(frame, gray, COLOR_BGR2GRAY); // convert image to grayscale
	equalizeHist(gray, gray); // enchance image contrast
	Mat blur;
	GaussianBlur(gray, blur, Size(9, 9), 0);
	Mat thresh;
	threshold(blur, thresh, 20, 255, THRESH_BINARY_INV);
	
	int upper_w = gray.cols;
	int upper_h = (int)((double)gray.rows * 0.60);
	Mat upper = thresh(Rect(0, 0, upper_w, upper_h));
	
	int histSize = 256;
    float range[] = { 0, 256 }; //the upper boundary is exclusive
    const float* histRange[] = { range };
    bool uniform = true, accumulate = false;
	Mat hist;
	calcHist(&upper, 1, 0, Mat(), hist, 1, &histSize, histRange, uniform, accumulate);

	prevFrameWhitePixelNo = currFrameWhitePixelNo;
	currFrameWhitePixelNo = hist.at<float>(255);
	float percentDiff = ((prevFrameWhitePixelNo - currFrameWhitePixelNo) / ((prevFrameWhitePixelNo + currFrameWhitePixelNo) / 2)) * 100;

	if (percentDiff >= 80.0) {
		close = true;
	} else if (percentDiff <= -80.0) {
		close = false;
	}

	frameNo += 1;
	if (blinkWindow == 0) {
		if (close) { 
			detectedBlinks[blinkCounter] = frameNo;
			blinkCounter += 1;
			printf("[%d] blink\n", frameNo);
			blinkWindow = 20;
		}
	} else {
		blinkWindow -= 1;
	}

	imshow("thresh", thresh);
	imshow("upper", upper);
}

int actualBlinks[30] = {1512,1573,1634,1693,1758,1810,1876,1935,1996,2052,2115,2172,2237,2294,2350,2407,2472,2537,2601,2655,2712,2764,2830,2884,2942,3001,3067,3127,3180,3237};

void getBlinkAccuracy() {
	int truePositive = 0, falsePositive = 0, falseNegative = 0;
	int window = 20;
	for (int i = 0; i < sizeof(detectedBlinks)/sizeof(detectedBlinks[0]); i++) {
		if (detectedBlinks[i] >= actualBlinks[0] - window && detectedBlinks[i] <= actualBlinks[30-1] + window) {
			for (int j = 0; j < sizeof(actualBlinks)/sizeof(actualBlinks[0]); j++) {
				if (detectedBlinks[i] >= actualBlinks[j] - window && detectedBlinks[i] <= actualBlinks[j] + window) {
					if ((detectedBlinks[i] - detectedBlinks[i-1]) < 40) {
						falsePositive += 1;
					} else {
						truePositive += 1;
					}
					break;
				} else {
					if (j == sizeof(actualBlinks)/sizeof(actualBlinks[0]) - 1) {
						falsePositive += 1;
					}
				}
			}
		}
	}

	falseNegative = sizeof(actualBlinks)/sizeof(actualBlinks[0]) - truePositive;
	double detectionRate = (double)truePositive/(truePositive + falseNegative) * 100;
	double falseAlarmRate = (double)falsePositive/(truePositive + falsePositive) * 100;
	double successRate = (double)detectionRate/(detectionRate + falseAlarmRate) * 100;
	printf("TP: %d FP: %d FN: %d DR: %.2f FAR: %.2f Success Rate: %.2f\n", truePositive, falsePositive, falseNegative, detectionRate, falseAlarmRate, successRate);
}

/* Main */
int main() {
	CascadeClassifier eyeCascade;
	if (!eyeCascade.load("/home/pi/Desktop/opencv-test/haarcascade/haarcascade_eye_tree_eyeglasses.xml")) {
        std::cerr << "Could not load eye detector." << std::endl;
        return -1;
    }
	VideoCapture cap("/home/pi/Desktop/opencv-test/vid/0x 15y.mp4");
	Mat frame;
	
	cap.read(frame);
	if (frame.empty()) return -1;
	gammaCorrection(frame, frame, 5);
	detectEyes(frame, eyeCascade);

	while (true) {
		cap.read(frame); // read stored frame
		if (frame.empty()) break;

		if (frameNo == actualBlinks[0] - 20) {
			gammaCorrection(frame, frame, 5);
			detectEyes(frame, eyeCascade);
		}

		frame = frame(eyes[0]);
		gammaCorrection(frame, frame, 1.5);
		detectBlink(frame);
		imshow("frame", frame);	
		if (waitKey(1) == 27) {
            cout << "Program terminated." << endl;
            break;
        }
	}

	getBlinkAccuracy();

	cap.release();
	destroyAllWindows();
	return 0;
}
