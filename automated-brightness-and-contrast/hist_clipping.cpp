
#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <time.h>
#include <vector>

using namespace cv;
using namespace std;

/* Main */
int main() {
	Mat frame, gray, new_frame;
	string path = "eye-1.png";
	frame = imread(path);
	cvtColor(frame, gray, COLOR_BGR2GRAY); // convert to grayscale

	// Calculate grayscale histogram
	Mat hist, accumulatedHist;
	int histSize = 256;
	float range[] = { 0, 256 }; //the upper boundary is exclusive
    const float* histRange[] = { range };
	calcHist( &gray, 1, 0, Mat(), hist, 1, &histSize, histRange);
	accumulatedHist = hist.clone();

	// Calcualte cumulative distrubution from the histogram
	for (int i = 1; i < histSize; i++) {
		accumulatedHist.at<float>(i) += accumulatedHist.at<float>(i - 1);
	}

	// Locate points to clip
	float clip_hist_percent = 50; 
	float maximum = accumulatedHist.at<float>(histSize - 1);
	clip_hist_percent *= (maximum / 100.0);
	clip_hist_percent /= 2.0;
	cout << "clip_hist_percent: " << clip_hist_percent << endl;

	// Locate left cut
	int minimum_gray = 0;
	while (accumulatedHist.at<float>(minimum_gray) < clip_hist_percent) {
		minimum_gray += 1;
	}

	// Locate right cut
	int maximum_gray = (histSize - 1);
	while (accumulatedHist.at<float>(maximum_gray) >= (maximum - clip_hist_percent)) {
		maximum_gray -= 1;
	}

	// Calculate alpha and beta values
	double alpha = 255 / (maximum_gray - minimum_gray);
	double beta_ = (-minimum_gray * alpha);
	cout << "alpha: " << alpha << endl;
	cout << "beta: " << beta_ << endl;

	convertScaleAbs(gray, new_frame, alpha, beta_);
	cout << "gray channels:" << gray.channels() << endl; 
	// for( int y = 0; y < gray.rows; y++ ) {
    //     for( int x = 0; x < gray.cols; x++ ) {
    //         for( int c = 0; c < gray.channels(); c++ ) {
    //             new_frame.at<Vec3b>(y,x)[c] = saturate_cast<uchar>( alpha*gray.at<Vec3b>(y,x)[c] + beta_ );
    //         }
    //     }
    // }
	// frame.convertTo(new_frame, -1, alpha, 0);

	imshow("Frame", frame);
	imshow("New Frame1", new_frame);
	equalizeHist(new_frame, new_frame);
	imshow("New Frame2", new_frame);


	waitKey(0);
	return 0;
}
