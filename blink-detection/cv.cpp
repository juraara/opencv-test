
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

void cropSides(Mat &src, Mat &dst, double crop_percent) {
	int x = src.cols * crop_percent;
	int y = src.rows * crop_percent;
	int src_w = src.cols * (1 - (crop_percent * 2));
	int src_h = src.rows * (1 - (crop_percent * 2));
	dst = src(Rect(x, y, src_w, src_h));
}

int frameNo = 0;
int blinkWindow = 0;

int detectedBlinks[255] = { 0 }; // all elements 0
int blinkCounter = 0;

RNG rng(12345);

void detectBlink(Mat &frame) {
	Mat gray;
	cvtColor(frame, gray, COLOR_BGR2GRAY); // convert image to grayscale
	equalizeHist(gray, gray); // enchance image contrast
	Mat blur;
	GaussianBlur(gray, blur, Size(9, 9), 0);
	Mat thresh;
	threshold(blur, thresh, 20, 255, THRESH_BINARY_INV);
	Mat crop;
	cropSides(thresh, crop, 0.2);

	Mat upper = crop(Rect(0, 0, crop.cols, (int)((double)crop.rows * 0.50)));

	vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;
	findContours(upper, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE);

	Mat drawing = Mat::zeros( upper.size(), CV_8UC3 );
    for (size_t i = 0; i < contours.size(); i++) {
        Scalar color = Scalar( rng.uniform(0, 256), rng.uniform(0,256), rng.uniform(0,256) );
        drawContours( drawing, contours, (int)i, color, 2, LINE_8, hierarchy, 0 );
    }

	frameNo += 1;
	if (blinkWindow == 0) {
		if (contours.size() == 0) { 
			detectedBlinks[blinkCounter] = frameNo;
			blinkCounter += 1;
			printf("[%d] blink\n", frameNo);
			blinkWindow = 20;
		}
	} else {
		blinkWindow -= 1;
	}

	imshow("upper", upper);
	imshow("drawing", drawing);
}

int countNonzero(int arr[], int size) {
	int count = 0;
	for (int i = 0; i < size; i++) {
		if (arr[i]) {
			count++;
		}
	}
	return count;
}

bool isBetween(int val, int start, int end) {
	return val >= start && val <= end;
}

void getBlinkAccuracy() {
	// int actualBlinks[30] = {15,75,135,195,255,315,375,435,495,555,615,675,735,795,855,915,975,1035,1095,1155,1215,1275,1335,1395,1455,1515,1575,1635,1695,1755};
	// int actualBlinks[30] = {8,68,128,188,248,308,368,428,488,548,608,668,728,788,848,908,968,1028,1088,1148,1208,1268,1328,1388,1448,1508,1568,1628,1688,1748};
	// int actualBlinks[30] = {27,87,147,207,267,327,387,447,507,567,627,687,747,807,867,927,987,1047,1107,1167,1227,1287,1347,1407,1467,1527,1587,1647,1707,1767};
	// int actualBlinks[30] = {11,71,131,191,251,311,371,431,491,551,611,671,731,791,851,911,971,1031,1091,1151,1211,1271,1331,1391,1451,1511,1571,1631,1691,1751};
	// int actualBlinks[30] = {10,70,130,190,250,310,370,430,490,550,610,670,730,790,850,910,970,1030,1090,1150,1210,1270,1330,1390,1450,1510,1570,1630,1690,1750};
	// int actualBlinks[30] = {20,80,140,200,260,320,380,440,500,560,620,680,740,800,860,920,980,1040,1100,1160,1220,1280,1340,1400,1460,1520,1580,1640,1700,1760};
	// int actualBlinks[30] = {33,93,153,213,273,333,393,453,513,573,633,693,753,813,873,933,993,1053,1113,1173,1233,1293,1353,1413,1473,1533,1593,1653,1713,1773};
	// int actualBlinks[30] = {71,131,191,251,311,371,431,491,551,611,671,731,791,851,911,971,1031,1091,1151,1211,1271,1331,1391,1451,1511,1571,1631,1691,1751,1811};
	int actualBlinks[30] = {27,87,147,207,267,327,387,447,507,567,627,687,747,807,867,927,987,1047,1107,1167,1227,1287,1347,1407,1467,1527,1587,1647,1707,1767};

	int truePositive = 0, falsePositive = 0, falseNegative = 0;
	int window = 20;

	for (int i = 0; i < countNonzero(detectedBlinks, 255); i++) {
		if (isBetween(detectedBlinks[i], actualBlinks[0] - window, actualBlinks[30-1] + window)) {
			for (int j = 0; j < countNonzero(actualBlinks, 30); j++) {
				if (isBetween(detectedBlinks[i], actualBlinks[j] - window, actualBlinks[j] + window)) {
					if (isBetween((detectedBlinks[i] - detectedBlinks[i-1]), window + 1, 60 - window)) { falsePositive += 1; } 
					else { truePositive += 1; }
					break;
				} 
				else if (j == countNonzero(actualBlinks, 30) - 1) {
					falsePositive += 1;
				}
			}
		}
	}

	/* Calculate */
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
	// VideoCapture cap("/home/pi/Desktop/opencv-test/vid/0x 15y.mp4");
	// VideoCapture cap("/home/pi/Desktop/opencv-test/vid/0x -15y.mp4");
	// VideoCapture cap("/home/pi/Desktop/opencv-test/vid/0x -45y.mp4");
	// VideoCapture cap("/home/pi/Desktop/opencv-test/vid/75x 15y.mp4");
	// VideoCapture cap("/home/pi/Desktop/opencv-test/vid/75x -15y.mp4");
	// VideoCapture cap("/home/pi/Desktop/opencv-test/vid/75x -45y.mp4");
	// VideoCapture cap("/home/pi/Desktop/opencv-test/vid/-75x 15y.mp4");
	// VideoCapture cap("/home/pi/Desktop/opencv-test/vid/-75x -15y.mp4");
	VideoCapture cap("/home/pi/Desktop/opencv-test/vid/-75x -45y.mp4");
	Mat frame;
	
	cap.read(frame);
	if (frame.empty()) return -1;
	gammaCorrection(frame, frame, 20);
	detectEyes(frame, eyeCascade);

	while (true) {
		cap.read(frame); // read stored frame
		if (frame.empty()) break;
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
