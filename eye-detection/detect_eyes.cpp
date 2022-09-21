#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <iostream>
#include <vector>

using namespace cv;
using namespace std;

int main() {
	/* Image */
	string imgDirectory10cm = "/home/pi/Desktop/opencv-test/img/10cm/";
	string imgDirectory20cm = "/home/pi/Desktop/opencv-test/img/20cm/";
	string imgDirectory30cm = "/home/pi/Desktop/opencv-test/img/30cm/";
	string imgFormat = ".png";
	string imgName[13] = {"90", "75", "60", "45", "30", "15", "00", "-15", "-30", "-45", "-60", "-75", "-90"};
	string imgPath10cm[13];
	string imgPath20cm[13];
	string imgPath30cm[13];
	
	for(int i = 0; i < 13; i++) {
		imgPath10cm[i] = imgDirectory10cm + imgName[i] + imgFormat;
		imgPath20cm[i] = imgDirectory20cm + imgName[i] + imgFormat;
		imgPath30cm[i] = imgDirectory30cm + imgName[i] + imgFormat;
	}

	Mat frame, crop, gray; // store frames here
	int imgNo = 0;
	
	/* Eye Detection */
	CascadeClassifier faceCascade;
	vector<Rect> eyes;
	faceCascade.load("/home/pi/Desktop/opencv-test/haarcascade/haarcascade_eye_tree_eyeglasses.xml");
    int faceDetectDelay = 0;

	/* 10 cm */
	cout << "Testing (10cm)" << endl;
	for (int i = 0; i < 13; i++) {
		frame = imread(imgPath10cm[i]);
		if (frame.empty()) break;
		cvtColor(frame, gray, COLOR_BGR2GRAY);
		
		vector<Rect> eyes;
		faceCascade.detectMultiScale(gray, eyes, 1.1, 10);
		/* Draw Rectangle Around the Eyes */
		for (int i = 0; i < eyes.size(); i++) {
			rectangle(gray, eyes[i].tl(), eyes[i].br(), Scalar(255, 255, 255), 2);
		}

		if (eyes.size()) {
			if (eyes.size() > 2) {
				cout << "Angle: " << imgName[i];
				cout << " TP: 1 FP: 1 FN: 0";
				cout << endl;
			} else {
				cout << "Angle: " << imgName[i];
				cout << " TP: 1 FP: 0 FN: 0";
				cout << endl;
			}
		} else {
			cout << "Angle: " << imgName[i];
			cout << " TP: 0 FP: 0 FN: 1";
			cout << endl;
		}

		imshow("Grayscale", gray);
		if (waitKey(1) == 27) {
			cout << "Program terminated." << endl;
			break;
		}
	}
	/* 20 cm */	
	cout << "\nTesting (20cm)" << endl;
	for (int i = 0; i < 13; i++) {
		frame = imread(imgPath20cm[i]);
		if (frame.empty()) break;
		cvtColor(frame, gray, COLOR_BGR2GRAY);
		
		vector<Rect> eyes;
		faceCascade.detectMultiScale(gray, eyes, 1.1, 10);
		/* Draw Rectangle Around the Eyes */
		for (int i = 0; i < eyes.size(); i++) {
			rectangle(gray, eyes[i].tl(), eyes[i].br(), Scalar(255, 255, 255), 2);
		}

		if (eyes.size()) {
			if (eyes.size() > 2) {
				cout << "Angle: " << imgName[i];
				cout << " TP: 1 FP: 1 FN: 0";
				cout << endl;
			} else {
				cout << "Angle: " << imgName[i];
				cout << " TP: 1 FP: 0 FN: 0";
				cout << endl;
			}
		} else {
			cout << "Angle: " << imgName[i];
			cout << " TP: 0 FP: 0 FN: 1";
			cout << endl;
		}

		imshow("Grayscale", gray);
		if (waitKey(1) == 27) {
			cout << "Program terminated." << endl;
			break;
		}
	}
	/* 30 cm */	
	cout << "\nTesting (30cm)" << endl;
	for (int i = 0; i < 13; i++) {
		frame = imread(imgPath30cm[i]);
		if (frame.empty()) break;
		cvtColor(frame, gray, COLOR_BGR2GRAY);
		
		vector<Rect> eyes;
		faceCascade.detectMultiScale(gray, eyes, 1.1, 10);
		/* Draw Rectangle Around the Eyes */
		for (int i = 0; i < eyes.size(); i++) {
			rectangle(gray, eyes[i].tl(), eyes[i].br(), Scalar(255, 255, 255), 2);
		}

		if (eyes.size()) {
			if (eyes.size() > 2) {
				cout << "Angle: " << imgName[i];
				cout << " TP: 1 FP: 1 FN: 0";
				cout << endl;
			} else {
				cout << "Angle: " << imgName[i];
				cout << " TP: 1 FP: 0 FN: 0";
				cout << endl;
			}
		} else {
			cout << "Angle: " << imgName[i];
			cout << " TP: 0 FP: 0 FN: 1";
			cout << endl;
		}

		imshow("Grayscale", gray);
		if (waitKey(1) == 27) {
			cout << "Program terminated." << endl;
			break;
		}
	}

	// cap.release();
	destroyAllWindows();
	return 0;
}

