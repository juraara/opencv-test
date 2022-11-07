#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <time.h>

using namespace cv;
using namespace std;

RNG rng(12345);

class BlinkDetectionAlgorithm {
    public: int frameNo = 0;
    public: int blinkWindow = 0;
    public: int detectedBlinks[255] = { 0 };
    public: int blinkCounter = 0;
    public: int window = 20;

    /* Gamma Correction */
    public: void gammaCorrection(const Mat &src, Mat &dst, const float gamma) {
        Mat gray;
        cvtColor(src, gray, COLOR_BGR2GRAY);

        float invGamma = 1 / gamma;
        Mat table(1, 256, CV_8U);
        uchar *p = table.ptr();
        for (int i = 0; i < 256; ++i) {
            p[i] = (uchar) (pow(i / 255.0, invGamma) * 255);
        }
        dst = src.clone();
        LUT(src, table, dst);
    }
    
    public: Rect getLeftmostEye(std::vector<cv::Rect> &eyes)
    {
        int leftmost = 99999999;
        int leftmostIndex = -1;
        for (int i = 0; i < eyes.size(); i++)
        {
            if (eyes[i].tl().x < leftmost)
            {
                leftmost = eyes[i].tl().x;
                leftmostIndex = i;
            }
        }
        return eyes[leftmostIndex];
    }

    /* Crop Detected Eyes */
    public: Rect detectEyes(Mat &frame, CascadeClassifier &eyeCascade) {
        Mat gray;
        gammaCorrection(frame, frame, 5); // brighten image
        cvtColor(frame, gray, COLOR_BGR2GRAY); // convert image to grayscale
        equalizeHist(gray, gray); // enchance image contrast

        std::vector<cv::Rect> eyes;
        eyeCascade.detectMultiScale(gray, eyes, 1.1, 2, 0 | CASCADE_SCALE_IMAGE, cv::Size(90, 90));
        if (eyes.size() == 0) {
            // no eye was detectedBlinks
            return Rect(0, 0, 0, 0);
        }
        for (cv::Rect &eye : eyes) {
            rectangle(frame, eye.tl(), eye.br(), cv::Scalar(0, 255, 0), 2);
        }
        
        Rect leftMostEye = getLeftmostEye(eyes);
        // printf("Height=%d Width=%d\n", leftMostEye.height, leftMostEye.width);
        rectangle(frame, leftMostEye.tl(), leftMostEye.br(), cv::Scalar(0, 0, 255), 2);
        imshow("frame", frame);
        return getLeftmostEye(eyes);
    }
    
    /* Calculate Blink Accuracy */
    public: void getBlinkAccuracy(int detectedBlinks[255], int ref[255]) {
        int truePositive = 0, falsePositive = 0, falseNegative = 0;

        int size1 = countNonzero(detectedBlinks, 255);
        int size2 = countNonzero(ref, 255);

        for (int i = 0; i < size1; i++) {
            if (isBetween(detectedBlinks[i], 7200, 10750)) {
                for (int j = 0; j < size2; j++) {
                    if (isBetween(detectedBlinks[i], ref[j] - window/2, ref[j] + window/2)) {
                        truePositive += 1;
                        break;
                    } 
                    else if (j == size2 - 1) {
                        falsePositive += 1;
                    }
                }
            }
        }

        falseNegative = size2 - truePositive;
        double detectionRate = (double)truePositive/(truePositive + falseNegative) * 100;
        double falseAlarmRate = (double)falsePositive/(truePositive + falsePositive) * 100;
        double criticalSuccessIndex = (double)truePositive/(truePositive + falsePositive + falseNegative) * 100;
        printf("TP: %d FP: %d FN: %d DR: %.2f FAR: %.2f CSI(TS): %.2f\n", truePositive, falsePositive, falseNegative, detectionRate, falseAlarmRate, criticalSuccessIndex);
    }

    /* Util */
    public: int countNonzero(int arr[], int size) {
        int count = 0;
        for (int i = 0; i < size; i++) {
            if (arr[i]) {
                count++;
            }
        }
        return count;
    }

    /* Util */
    public: bool isBetween(int val, int start, int end) {
        return val >= start && val <= end;
    }
};

class ContourDetectionAndAnalysis: public BlinkDetectionAlgorithm {
    
    public: void detectBlink(Mat &frame) {
        gammaCorrection(frame, frame, 5);
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

        Mat upper = crop(Rect(0, 0, crop.cols, (int)((double)crop.rows * 0.50)));

        vector<vector<Point>> contours;
        vector<Vec4i> hierarchy;
        findContours(upper, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE);

        Mat drawing = Mat::zeros( upper.size(), CV_8UC3 );
        for (size_t i = 0; i < contours.size(); i++) {
            Scalar color = Scalar( rng.uniform(0, 256), rng.uniform(0,256), rng.uniform(0,256) );
            drawContours( drawing, contours, (int)i, color, 2, LINE_8, hierarchy, 0 );
        }

        Point p1(x, y);
        Point p2(x+src_h, y+src_w);
        rectangle(gray, p1, p2, cv::Scalar(0, 255, 0), 2);
        line(gray, Point(x, y+src_h/2), Point(x+src_w, y+src_h/2), Scalar(0, 0, 255), 2, 8, 0);
        imshow("gray", gray);
        imshow("crop", crop);
        imshow("upper", upper);
        imshow("drawing", drawing);

        frameNo += 1;
        if (frameNo < 7200) return;
        if (blinkWindow == 0) {
            if (contours.size() == 0) { 
                detectedBlinks[blinkCounter] = frameNo;
                blinkCounter += 1;
                // printf("[%d] blink\n", frameNo);
                blinkWindow = window;
            }
        } else {
            blinkWindow -= 1;
        }
        
    }
};

class GlobalThresholdingOfNegative: public BlinkDetectionAlgorithm {
    private: float prevFrameWhitePixelNo = 0.0;
    private: float currFrameWhitePixelNo = 0.0;
    private: bool close = false; 

    public: void detectBlink(Mat &frame) {
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
        int upper_h = (int)((double)crop.rows * 0.50);
        Mat upper = crop(Rect(0, 0, upper_w, upper_h));
        
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
        } else if (percentDiff <= -20.0) {
            close = false;
        }

        Point p1(x, y);
        Point p2(x+src_h, y+src_w);
        rectangle(gray, p1, p2, cv::Scalar(0, 255, 0), 2);
        line(gray, Point(x, y+src_h/2), Point(x+src_w, y+src_h/2), Scalar(0, 0, 255), 2, 8, 0);
        imshow("gray", gray);
        imshow("crop", crop);
	    imshow("upper", upper);

        frameNo += 1;
        if (frameNo < 7200) return;
        if (blinkWindow == 0) {
            if (close) { 
                detectedBlinks[blinkCounter] = frameNo;
                blinkCounter += 1;
                // printf("[%d] blink\n", frameNo);
                blinkWindow = window;
            }
        } else {
            blinkWindow -= 1;
        }
    }
};

class EyeBlackPixelRatioAnalysis: public BlinkDetectionAlgorithm {
    public: void detectBlink(Mat &frame) {
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
        int upper_h = (int)((double)crop.rows * 0.60);
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

        Point p1(x, y);
        Point p2(x+src_h, y+src_w);
        rectangle(gray, p1, p2, cv::Scalar(0, 255, 0), 2);
        line(gray, Point(x, y+src_h/2), Point(x+src_w, y+src_h/2), Scalar(0, 0, 255), 2, 8, 0);
        imshow("gray", gray);
        imshow("crop", crop);
        imshow("upper", upper);
        imshow("lower", lower);

        frameNo += 1;
        if (frameNo < 7200) return;
        if (blinkWindow == 0) {
            if (whitePixelNo_up < whitePixelNo_low) { 
                detectedBlinks[blinkCounter] = frameNo;
                blinkCounter += 1;
                // printf("[%d] blink\n", frameNo);
                blinkWindow = window;
            }
        } else {
            blinkWindow -= 1;
        }

    }
};

class QuickSortAndAnalysis_7200: public BlinkDetectionAlgorithm {
    /* Partition */
    public: int partition(float arr[], int start, int end) {
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

    public: void quickSort(float arr[], int start, int end) {
        if (start >= end) return; // base case
        int p = partition(arr, start, end); // partitioning the array
        quickSort(arr, start, p - 1); // Sorting the left part
        quickSort(arr, p + 1, end); // Sorting the right part
    }

    public: void rotateArr(float arr[], int d, int n) {
        int p = 1;
        while (p <= d) {
            int last = arr[0];
            for (int i = 0; i < n - 1; i++) {
                arr[i] = arr[i + 1];
            }
            arr[n - 1] = 0;
            p++;
        }
    }

    public: int windowSize;
    public: float blackPixel[7200] = { 0 };
    
    private: int counter = 0;
    private: int close = 0, open = 0;
    private: float PERCLOS = 0, PERCLOS_old = 0;
    private: int temp = 0;

    public: QuickSortAndAnalysis_7200(int w) {
        windowSize = w;
    }

    public: void detectBlink(Mat &frame) {
        gammaCorrection(frame, frame, 5);
        Mat gray;
        cvtColor(frame, gray, COLOR_BGR2GRAY); // convert image to grayscale
        equalizeHist(gray, gray); // enchance image contrast
        Mat blur;
        GaussianBlur(gray, blur, Size(9, 9), 0);
        Mat thresh;
        threshold(blur, thresh, 20, 255, THRESH_BINARY);
        
        double crop_percent = 0.2;
        int x = thresh.cols * crop_percent;
        int y = thresh.rows * crop_percent;
        int src_w = thresh.cols * (1 - (crop_percent * 2));
        int src_h = thresh.rows * (1 - (crop_percent * 2));
        Mat crop = thresh(Rect(x, y, src_w, src_h));
        Mat upper = crop(Rect(0, 0, crop.cols, (int)((double)crop.rows * 0.55)));
        
        int histSize = 256;
        float range[] = { 0, 256 }; //the upper boundary is exclusive
        const float* histRange[] = { range };
        bool uniform = true, accumulate = false;
        Mat hist;
        calcHist(&upper, 1, 0, Mat(), hist, 1, &histSize, histRange, uniform, accumulate);

        // if (counter < windowSize - 1) {
        //     printf("[%d] Storing\n", counter);
        // }

        blackPixel[counter] = hist.at<float>(0);
        // cout << blackPixel[counter] << endl;
        counter++;

        if (counter == windowSize) {
            counter = windowSize - 1;

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
            close = 0; open = 0;
            for (int i = 0; i < windowSize; i++) {
                if (sorted[i] < threshold) { close++; } 
                else { open++; }
            }

            /* Time Proportion */
            PERCLOS_old = PERCLOS;
            PERCLOS = (float)close / (close + open) * 100;
            // printf("PERCLOS=%.2f\n", PERCLOS);

            /* SHL Array */
            rotateArr(blackPixel, 1, 7200);
        }

        Point p1(x, y);
        Point p2(x+src_h, y+src_w);
        rectangle(gray, p1, p2, cv::Scalar(0, 255, 0), 2);
        line(gray, Point(x, y+src_h/2), Point(x+src_w, y+src_h/2), Scalar(0, 0, 255), 2, 8, 0);
        imshow("gray", gray);
        imshow("crop", crop);
        imshow("upper", upper);

        frameNo += 1;
        if (frameNo < 7200) return;
        if (blinkWindow == 0) {
            if (PERCLOS_old < PERCLOS) { 
                detectedBlinks[blinkCounter] = frameNo;
                blinkCounter += 1;
                // printf("[%d] Blink\n", frameNo);
                blinkWindow = window;
            }
        } 
        else {
            blinkWindow -= 1;
        }

    }
};

class QuickSortAndAnalysis_3600: public QuickSortAndAnalysis_7200 {
    public: QuickSortAndAnalysis_3600(int w) : QuickSortAndAnalysis_7200(w) {
    }
};

class QuickSortAndAnalysis_1800: public QuickSortAndAnalysis_7200 {
    public: QuickSortAndAnalysis_1800(int w): QuickSortAndAnalysis_7200(w) {
    }
};

class MyClass: public BlinkDetectionAlgorithm {
    public: void myMethod(string path, int ref[255], int code) {
        ContourDetectionAndAnalysis obj1;
        GlobalThresholdingOfNegative obj2;
        EyeBlackPixelRatioAnalysis obj3;
        QuickSortAndAnalysis_7200 obj4(7200);
        QuickSortAndAnalysis_3600 obj5(3600);
        QuickSortAndAnalysis_1800 obj6(1800);

        CascadeClassifier eyeCascade;
        if (!eyeCascade.load("/home/pi/Desktop/opencv-test/res/haarcascade/haarcascade_eye_tree_eyeglasses.xml")) {
            std::cerr << "Could not load eye detector." << std::endl;
            return;
        }

        VideoCapture cap(path);
        Mat frame;
        
        cap.read(frame);
        if (frame.empty()) return;
        Rect eyeRIO = detectEyes(frame, eyeCascade);

        while (true) {
            cap.read(frame);
            if (frame.empty()) break;
            if (eyeRIO.empty()) break;
            frame = frame(eyeRIO);
            
            switch (code) {
                case 1: obj1.detectBlink(frame); break;
                case 2: obj2.detectBlink(frame); break;
                case 3: obj3.detectBlink(frame); break;
                case 4: obj4.detectBlink(frame); break;
                case 5: obj5.detectBlink(frame); break;
                default: obj6.detectBlink(frame); break;
            }

            if (waitKey(1) == 27) {
                cout << "Program terminated." << endl;
                break;
            }
        }
        switch (code) {
            case 1: getBlinkAccuracy(obj1.detectedBlinks, ref); break;
            case 2: getBlinkAccuracy(obj2.detectedBlinks, ref); break;
            case 3: getBlinkAccuracy(obj3.detectedBlinks, ref); break;
            case 4: getBlinkAccuracy(obj4.detectedBlinks, ref); break;
            case 5: getBlinkAccuracy(obj5.detectedBlinks, ref); break;
            default: getBlinkAccuracy(obj6.detectedBlinks, ref); break;
        }
        
        if (getWindowProperty("frame", WND_PROP_VISIBLE) > -1) destroyWindow("frame");
        if (getWindowProperty("gray", WND_PROP_VISIBLE) > -1) destroyWindow("gray");
        if (getWindowProperty("crop", WND_PROP_VISIBLE) > -1) destroyWindow("crop");
        if (getWindowProperty("upper", WND_PROP_VISIBLE) > -1) destroyWindow("upper");
        if (getWindowProperty("lower", WND_PROP_VISIBLE) > -1) destroyWindow("lower");
       if (getWindowProperty("drawing", WND_PROP_VISIBLE) > -1) destroyWindow("drawing");
	}
};

int main() {
    int ref[3][255] = {{7246,7278,7340,7506,7685,7850,7917,8019,8154,8186,8288,8415,8533,8690,8821,8953,9055,9201,9296,9319,9466,9509,9611,9733,9821,9891,9937,10035,10090,10254,10457,10579,10742}, {9852,9955,10291}, {7722,7775,7821,7880,7959,8062,8238,8486,8654,8771,8846,8930,9057,9220,9340,9478,9675,9813,9870,9978,10107,10228,10418,10599}};
    string path[3] = {"/home/pi/Desktop/opencv-test/res/vid/Test-01.mp4", "/home/pi/Desktop/opencv-test/res/vid/Test-02.mp4", "/home/pi/Desktop/opencv-test/res/vid/Test-03.mp4"};
    
    // cout << "Blink Detection Accuracy Comparison Testing (1/4)" << endl;
    // cout << "Contour Detection and Analysis" << endl;
    // MyClass contourDetectionAndAnalysis[9];
    // for (int i = 0; i < 3; i++) {
    //     cout << "(" << i+1 << "/3): "; 
    //     contourDetectionAndAnalysis[i].myMethod(path[i], ref[i], 1);
    // }
    
    // cout << endl;
    // cout << "Blink Detection Accuracy Comparison Testing (2/4)" << endl;
    // cout << "Global Thresholding of Negative" << endl;
    // MyClass globalThresholdingOfNegative[9];
    // for (int i = 0; i < 3; i++) {
    //     cout << "(" << i+1 << "/3): "; 
    //     globalThresholdingOfNegative[i].myMethod(path[i], ref[i], 2);
    // }

    // cout << endl;
    // cout << "Blink Detection Accuracy Comparison Testing (3/4)" << endl;
    // cout << "Eye Black Pixel Ratio Analysis" << endl;
    // MyClass eyeBlackPixelRatioAnalysis[9];
    // for (int i = 0; i < 3; i++) {
    //     cout << "(" << i+1 << "/3): "; 
    //     eyeBlackPixelRatioAnalysis[i].myMethod(path[i], ref[i], 3);
    // }
    
    // cout << endl;
    // cout << "Blink Detection Accuracy Comparison Testing (4/4)" << endl;
    // cout << "Quick Sort and Analysis (120s Loading Time)" << endl;
    // MyClass quickSortAndAnalysis_7200[9];
    // for (int i = 0; i < 3; i++) {
    //     cout << "(" << i+1 << "/3): "; 
    //     quickSortAndAnalysis_7200[i].myMethod(path[i], ref[i], 4);
    // }

    // cout << "End of Blink Detection Accuracy Comparison Test." << endl;

    // cout << "Loading Time Duration Comparison Testing (1/2)" << endl;
    // cout << "Quick Sort and Analysis (60s Loading Time)" << endl;
    // MyClass quickSortAndAnalysis_3600[9];
    // for (int i = 0; i < 3; i++) {
    //     cout << "(" << i+1 << "/3): "; 
    //     quickSortAndAnalysis_3600[i].myMethod(path[i], ref[i], 5);
    // }

    cout << "Loading Time Duration Comparison Testing (2/2)" << endl;
    cout << "Quick Sort and Analysis (30s Loading Time)" << endl;
    MyClass quickSortAndAnalysis_1800[9];
    for (int i = 0; i < 3; i++) {
        cout << "(" << i+1 << "/3): "; 
        quickSortAndAnalysis_1800[i].myMethod(path[i], ref[i], 6);
    }

    cout << "End of Test." << endl;

    return 0;
}