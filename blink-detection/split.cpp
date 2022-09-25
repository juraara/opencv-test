
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
	// equalizeHist(gray, gray); // enchance image contrast

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

int detectedBlinks[255];
int blinkCounter = 0;

void detectBlink(Mat &frame) {
	Mat gray;
	cvtColor(frame, gray, COLOR_BGR2GRAY); // convert image to grayscale
	equalizeHist(gray, gray); // enchance image contrast
	Mat blur;
	GaussianBlur(gray, blur, Size(9, 9), 0);
	Mat thresh;
	threshold(blur, thresh, 20, 255, THRESH_BINARY_INV);

	double crop_percent = 0.2;
	int x = thresh.cols * crop_percent;
	int y = thresh.rows * crop_percent;
	int src_w = thresh.cols * (1 - (crop_percent * 2));
	int src_h = thresh.rows * (1 - (crop_percent * 2));
	Mat crop = thresh(Rect(x, y, src_w, src_h));

	int upper_w = crop.cols;
	int upper_h = (int)((double)crop.rows * 0.55);
	int lower_w = upper_w;
	int lower_h = crop.rows - upper_h;
	Mat upper = crop(Rect(0, 0, upper_w, upper_h));
	Mat lower = crop(Rect(0, upper_h, lower_w, lower_h));
	
	int histSize = 256;
    float range[] = { 0, 256 }; //the upper boundary is exclusive
    const float* histRange[] = { range };
    bool uniform = true, accumulate = false;
	Mat hist_up, hist_low;
	calcHist(&upper, 1, 0, Mat(), hist_up, 1, &histSize, histRange, uniform, accumulate);
	calcHist(&lower, 1, 0, Mat(), hist_low, 1, &histSize, histRange, uniform, accumulate);

	float whitePixelNo_up = hist_up.at<float>(255);
	float whitePixelNo_low = hist_low.at<float>(255);

	frameNo += 1;
	if (blinkWindow == 0) {
		if (whitePixelNo_up < whitePixelNo_low) { 
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
	imshow("lower", lower);
}

int actualBlinks[30] = {1495,1555,1615,1675,1735,1795,1855,1915,1975,2035,2095,2155,2215,2275,2335,2395,2455,2515,2575,2635,2695,2755,2815,2875,2935,2995,3055,3115,3175,3235};
// int actualBlinks[30] = {706,766,826,886,946,1006,1066,1126,1186,1246,1306,1366,1426,1486,1546,1606,1666,1726,1786,1846,1906,1966,2026,2086,2146,2206,2266,2326,2386,2446};
// int actualBlinks[30] = {478,538,598,658,718,778,838,898,958,1018,1078,1138,1198,1258,1318,1378,1438,1498,1558,1618,1678,1738,1798,1858,1918,1978,2038,2098,2158,2218};
// int actualBlinks[30] = {1171,1231,1291,1351,1411,1471,1531,1591,1651,1711,1771,1831,1891,1951,2011,2071,2131,2191,2251,2311,2371,2431,2491,2551,2611,2671,2731,2791,2851,2911};
// int actualBlinks[30] = {1640,1700,1760,1820,1880,1940,2000,2060,2120,2180,2240,2300,2360,2420,2480,2540,2600,2660,2720,2780,2840,2900,2960,3020,3080,3140,3200,3260,3320,3380};
// int actualBlinks[30] = {791,851,911,971,1031,1091,1151,1211,1271,1331,1391,1451,1511,1571,1631,1691,1751,1811,1871,1931,1991,2051,2111,2171,2231,2291,2351,2411,2471,2531};
// int actualBlinks[30] = {1512,1573,1634,1693,1758,1810,1876,1935,1996,2052,2115,2172,2237,2294,2350,2407,2472,2537,2601,2655,2712,2764,2830,2884,2942,3001,3067,3127,3180,3237};
// int actualBlinks[30] = {845,907,964,1023,1082,1144,1202,1264,1331,1386,1451,1508,1562,1622,1679,1737,1800,1865,1917,1983,2049,2098,2152,2212,2277,2330,2385,2450,2508,2567};
// int actualBlinks[30] = {516,573,630,687,750,812,859,934,997,1061,1121,1174,1234,1288,1346,1401,1462,1525,1585,1655,1724,1773,1827,1880,1932,1998,2061,2122,2179,2243};

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
	// VideoCapture cap("/home/pi/Desktop/opencv-test/vid/0x -15y.mp4");
	// VideoCapture cap("/home/pi/Desktop/opencv-test/vid/0x -45y.mp4");
	// VideoCapture cap("/home/pi/Desktop/opencv-test/vid/75x 15y.mp4");
	// VideoCapture cap("/home/pi/Desktop/opencv-test/vid/75x -15y.mp4");
	// VideoCapture cap("/home/pi/Desktop/opencv-test/vid/75x -45y.mp4");
	// VideoCapture cap("/home/pi/Desktop/opencv-test/vid/-75x 15y.mp4");
	// VideoCapture cap("/home/pi/Desktop/opencv-test/vid/-75x -15y.mp4");
	// VideoCapture cap("/home/pi/Desktop/opencv-test/vid/-75x -45y.mp4");
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
