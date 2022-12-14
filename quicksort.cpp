/* Image Processing Module
 * (Quicksort Analysis) 
 * 1. Performs well when there is a lot of data (more frames)
 * 2. Optimally this should be running for a couple of minutes to ensure reliable results according to Yan et. al. (Real-time Driver Drowsiness Detection System Based on PERCLOS and Grayscale Image Processing). 
 * Should it run less than the optimal duration, it may still perform well assuming the person provides adequate data for open and closed states. That is, the person is blinking frequently. */

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

/* Quicksort */
void quickSort(float arr[], int start, int end) {
    if (start >= end) return; // base case
    int p = partition(arr, start, end); // partitioning the array
    quickSort(arr, start, p - 1); // Sorting the left part
    quickSort(arr, p + 1, end); // Sorting the right part
}

int main() {
	/* Camera */
	VideoCapture cap(0); // default cam
	Mat frame, crop, gray, blur, thresh;
	int minThresh = 50; // for thresholding
	int maxThresh = 255; // for thresholding
	int frameNo = 0;
	
	/* Blink Util */
	float framePixels[65];
	int arraySize = 65;
	int count = 0;
	
	while(true) {
		clock_t start = lClock(); // start counting
		cap.read(frame);
	
		/* Process image */
		crop = frame(Rect(170, 180, 230, 140)); // crop frame
		cvtColor(crop, gray, COLOR_BGR2GRAY); // convert to grayscale
		imshow("Gray", gray);
		GaussianBlur(gray, blur, Size(9, 9), 0); // apply gaussian blur
		threshold(blur, thresh, minThresh, maxThresh, THRESH_BINARY); // apply thresholding
		imshow("Threshold", thresh); // display window
		
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
		
		calcHist(&thresh, 1, 0, Mat(), histogram, 1, &numberBins, &channelRanges); // calculate
		
		/* Store pixels per frame */
		framePixels[count] = histogram.at<float>(0);
		count++;
		if (count == 65) {
			count = 0;
			/* Quicksort */
			cout << endl;
			quickSort(framePixels, 0, arraySize - 1);
			for (int i = 0; i < arraySize; i++) {
				// cout << framePixels[i] << ", ";
			}
			
			/* Get Average Number of Pixels (Open State) */
			int openEnd = arraySize - ((double)arraySize / 20); 
			int openStart = arraySize - ((double)arraySize / 10);  
			float sumOpen = 0.0;
			int openStates = 0;
			for (int i = openStart; i <= openEnd; i++) {
				sumOpen += framePixels[i];
			}
			openStates = sumOpen / (arraySize / 20);
			cout << "Avg Open State Px.: " << openStates << endl;
			
			/* Get Average Number of Pixels (Closed State) */
			int closeEnd = arraySize / 10;
			int closeStart = arraySize / 20;
			float sumClose = 0.0;
			int closeStates = 0;
			for (int i = closeStart; i <= closeEnd; i++) {
				sumClose += framePixels[i];
			}
			closeStates = sumClose / (arraySize / 20);
			cout << "Avg. Closed State Px.: " << closeStates << endl; 

			/* Threshold P80 */
			float threshold = (openStates - closeStates) * 0.2 +  closeStates;

			/* Perform Racial Segregation */
			int close = 0;
			int open = 0;
			for (int i = 0; i < arraySize; i++) {
				if (framePixels[i] < threshold) {
					close++;
				} else {
					open++;
				}
			}
			cout << "Closed States: " << close << endl;
			cout << "Open States: " << open << endl;
			
			/* PERCLOS */
			double perclos = ((double)close / 65) * 100;
			cout << "Perclos: " << perclos << endl;
			
			/* Others */
			double duration = lClock() - start; // stop counting
			double averageTimePerFrame = averageDuration(duration); // avg time per frame
			cout << "Avg tpf: " << averageTimePerFrame << "ms" << endl;
			cout << "Avg fps: " << averageFps() << endl;
		}
		averageFps();
		
		/* Exit at esc key */
		if (waitKey(1) == 27) {
            cout << "Program terminated." << endl;
            break;
        }
	}
	
	return 0;
}
