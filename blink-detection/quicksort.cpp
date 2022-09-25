
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

/* Partition */
int partition(float arr[], int start, int end) {
    float pivot = arr[start];
    int count = 0;
    for (int i = start + 1; i <= end; i++) {
        if (arr[i] <= pivot)
            count++;
    }
    
    /* Giving pivot element its correct position */
    int pivotIndex = start + count;
    swap(arr[pivotIndex], arr[start]);

    /* Sorting left and right parts of the pivot element */
    int i = start, j = end;
    while (i < pivotIndex && j > pivotIndex) {
 
        while (arr[i] <= pivot) {
            i++;
        }
        while (arr[j] > pivot) {
            j--;
        }
        if (i < pivotIndex && j > pivotIndex) {
            swap(arr[i++], arr[j--]);
        }
    }
    return pivotIndex;
}

void quickSort(float arr[], int start, int end) {
    if (start >= end) return; // base case
    int p = partition(arr, start, end); // partitioning the array
    quickSort(arr, start, p - 1); // Sorting the left part
    quickSort(arr, p + 1, end); // Sorting the right part
}

void rotateArr(float arr[], int d, int n)
{
    int p = 1;
    while (p <= d) {
        int last = arr[0];
        for (int i = 0; i < n - 1; i++) {
            arr[i] = arr[i + 1];
        }
        arr[n - 1] = last;
        p++;
    }
}

const int windowSize = 1925;
float blackPixel[windowSize];

/* Populate Array */
void populateArr(string path, CascadeClassifier &eyeCascade) {
	int i = 0;
	VideoCapture cap(path);
	Mat frame;

	cap.read(frame);
	if (frame.empty()) return;
	gammaCorrection(frame, frame, 5);
	detectEyes(frame, eyeCascade);

	while (true) {
		cap.read(frame);
		if (frame.empty() || i == windowSize) {
			break;
		}
		frame = frame(eyes[0]);
		gammaCorrection(frame, frame, 1.5);
		
		Mat gray;
		cvtColor(frame, gray, COLOR_BGR2GRAY); // convert image to grayscale
		equalizeHist(gray, gray); // enchance image contrast
		Mat blur;
		GaussianBlur(gray, blur, Size(9, 9), 0);
		Mat thresh;
		threshold(blur, thresh, 20, 255, THRESH_BINARY);

		int upper_w = gray.cols;
		int upper_h = (int)((double)gray.rows * 0.50);
		Mat upper = thresh(Rect(0, 0, upper_w, upper_h));
		
		int histSize = 256;
		float range[] = { 0, 256 }; //the upper boundary is exclusive
		const float* histRange[] = { range };
		bool uniform = true, accumulate = false;
		Mat hist;
		calcHist(&upper, 1, 0, Mat(), hist, 1, &histSize, histRange, uniform, accumulate);

		blackPixel[i] = hist.at<float>(0);
		printf("[%d] black pixels: %.1f\n", i, blackPixel[i]);
		i++;

		imshow("upper", upper);
		if (waitKey(1) == 27) {
            cout << "Program terminated." << endl;
            break;
        }
	}

	for (int i = 0; i < windowSize; i++){
		cout << blackPixel[i] << " ";
	}
	cout << endl;
}

/* Blink Detection */
int frameNo = 0;
int blinkWindow = 0;

int detectedBlinks[50];
int blinkCounter = 0;

int counter = windowSize - 1;
int close = 0, open = 0;
int close_old = 0, open_old = 0;

void detectBlink(Mat &frame) {
	Mat gray;
	cvtColor(frame, gray, COLOR_BGR2GRAY); // convert image to grayscale
	equalizeHist(gray, gray); // enchance image contrast
	Mat blur;
	GaussianBlur(gray, blur, Size(9, 9), 0);
	Mat thresh;
	threshold(blur, thresh, 20, 255, THRESH_BINARY);
	
	int upper_w = gray.cols;
	int upper_h = (int)((double)gray.rows * 0.50);
	Mat upper = thresh(Rect(0, 0, upper_w, upper_h));
	
	int histSize = 256;
    float range[] = { 0, 256 }; //the upper boundary is exclusive
    const float* histRange[] = { range };
    bool uniform = true, accumulate = false;
	Mat hist;
	calcHist(&upper, 1, 0, Mat(), hist, 1, &histSize, histRange, uniform, accumulate);

	blackPixel[counter] = hist.at<float>(0);
	counter++;

	if (counter == windowSize) {
		counter = windowSize - 1;

		/* SHL Array */
		rotateArr(blackPixel, 1, windowSize);

		/* Quicksort */
		float sorted[windowSize];
		for (int i = 0; i < windowSize; i++) {
			sorted[i] = blackPixel[i];
		}
		quickSort(sorted, 0, windowSize - 1);
		
		/* Get Average Number of Pixels (Open State) */
		int openEnd = windowSize - ((double)windowSize / 20); 
		int openStart = windowSize - ((double)windowSize / 10);  
		float sumOpen = 0.0;
		int openStates = 0;
		for (int i = openStart; i <= openEnd; i++) {
			sumOpen += sorted[i];
		}
		openStates = sumOpen / (windowSize / 20);
		
		/* Get Average Number of Pixels (Closed State) */
		int closeEnd = windowSize / 10;
		int closeStart = windowSize / 20;
		float sumClose = 0.0;
		int closeStates = 0;
		for (int i = closeStart; i <= closeEnd; i++) {
			sumClose += sorted[i];
		}
		closeStates = sumClose / (windowSize / 20);
		
		/* Threshold P80 */
		float threshold = (openStates - closeStates) * 0.2 +  closeStates;

		/* Separate Open from Closed State */
		close_old = close; open_old = open;
		close = 0; open = 0;
		for (int i = 0; i < windowSize; i++) {
			if (blackPixel[i] < threshold) { close++; } 
			else { open++; }
		}
	}

	frameNo += 1;
	if (blinkWindow == 0) {
		if (close_old < close) { 
			detectedBlinks[blinkCounter] = frameNo;
			blinkCounter += 1;
			printf("[%d] blink\n", frameNo);
			blinkWindow = 20;
		}
	} else {
		blinkWindow -= 1;
	}

	imshow("upper", upper);
}

int actualBlinks[30] = {706,766,826,886,946,1006,1066,1126,1186,1246,1306,1366,1426,1486,1546,1606,1666,1726,1786,1846,1906,1966,2026,2086,2146,2206,2266,2326,2386,2446};

void getBlinkAccuracy() {
	int truePositive = 0, falsePositive = 0, falseNegative = 0;
	int window = 20;
	for (int i = 0; i < sizeof(detectedBlinks)/sizeof(detectedBlinks[0]); i++) {
		if (detectedBlinks[i] >= actualBlinks[0] - window && detectedBlinks[i] <= actualBlinks[30-1] + window) {
			for (int j = 0; j < sizeof(actualBlinks)/sizeof(actualBlinks[0]); j++) {
				if (detectedBlinks[i] >= actualBlinks[j] - window && detectedBlinks[i] <= actualBlinks[j] + window) {
					truePositive += 1;
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

	string path = "/home/pi/Desktop/opencv-test/vid/0x 15y.mp4";
	populateArr(path, eyeCascade);
	VideoCapture cap(path);
	Mat frame;
	
	cap.read(frame);
	if (frame.empty()) return -1;
	gammaCorrection(frame, frame, 5);
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
