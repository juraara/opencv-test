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
    
    /* Crop Detected Eyes */
    public: Rect detectEyes(Mat &frame, CascadeClassifier &eyeCascade) {
        Mat gray;
        gammaCorrection(frame, frame, 5); // brighten image
        cvtColor(frame, gray, COLOR_BGR2GRAY); // convert image to grayscale
        equalizeHist(gray, gray); // enchance image contrast

        std::vector<cv::Rect> eyes;
        eyeCascade.detectMultiScale(gray, eyes, 1.1, 2, 0 | CASCADE_SCALE_IMAGE, cv::Size(30, 30));
        if (eyes.size() == 0) {
            // no eye was detectedBlinks
            return Rect(0, 0, 0, 0);
        }
        for (cv::Rect &eye : eyes) {
            rectangle(frame, eye.tl(), eye.br(), cv::Scalar(0, 255, 0), 2);
        }
        rectangle(frame, eyes[0].tl(), eyes[0].br(), cv::Scalar(0, 0, 255), 2);
        
        // imshow("frame", frame);
        return eyes[0];
    }
    
    /* Calculate Blink Accuracy */
    public: void getBlinkAccuracy(int detectedBlinks[255], int ref[30]) {
        int truePositive = 0, falsePositive = 0, falseNegative = 0;
        int window = 20;

        for (int i = 0; i < countNonzero(detectedBlinks, 255); i++) {
            if (isBetween(detectedBlinks[i], ref[0] - window, ref[30-1] + window)) {
                for (int j = 0; j < countNonzero(ref, 30); j++) {
                    if (isBetween(detectedBlinks[i], ref[j] - window, ref[j] + window)) {
                        if (isBetween((detectedBlinks[i] - detectedBlinks[i-1]), window + 1, 60 - window)) { falsePositive += 1; } 
                        else { truePositive += 1; }
                        break;
                    } 
                    else if (j == countNonzero(ref, 30) - 1) {
                        falsePositive += 1;
                    }
                }
            }
        }

        falseNegative = countNonzero(ref, 30) - truePositive;
        double detectionRate = (double)truePositive/(truePositive + falseNegative) * 100;
        double falseAlarmRate = (double)falsePositive/(truePositive + falsePositive) * 100;
        double successRate = (double)detectionRate/(detectionRate + falseAlarmRate) * 100;
        printf("TP: %d FP: %d FN: %d DR: %.2f FAR: %.2f Success Rate: %.2f\n", truePositive, falsePositive, falseNegative, detectionRate, falseAlarmRate, successRate);
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

        frameNo += 1;
        if (blinkWindow == 0) {
            if (contours.size() == 0) { 
                detectedBlinks[blinkCounter] = frameNo;
                blinkCounter += 1;
                // printf("[%d] blink\n", frameNo);
                blinkWindow = 20;
            }
        } else {
            blinkWindow -= 1;
        }
        
        imshow("upper", upper);
        imshow("drawing", drawing);
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

        if (percentDiff >= 40.0) {
            close = true;
        } else if (percentDiff <= -40.0) {
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

        imshow("crop", crop);
	    imshow("upper", upper);
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

        imshow("crop", crop);
        imshow("upper", upper);
        imshow("lower", lower);
    }
};

class QuickSortAndAnalysis: public BlinkDetectionAlgorithm {
    /* Partition */
    private: int partition(float arr[], int start, int end) {
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

    private: void quickSort(float arr[], int start, int end) {
        if (start >= end) return; // base case
        int p = partition(arr, start, end); // partitioning the array
        quickSort(arr, start, p - 1); // Sorting the left part
        quickSort(arr, p + 1, end); // Sorting the right part
    }

    private: void rotateArr(float arr[], int d, int n) {
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

    private: void formatFrame(Mat &src, Mat &dst) {
        gammaCorrection(src, src, 5);
        Mat gray;
        cvtColor(src, gray, COLOR_BGR2GRAY); // convert image to grayscale
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

        int upper_w = crop.cols;
        int upper_h = (int)((double)crop.rows * 0.55);
        dst = crop(Rect(0, 0, upper_w, upper_h));
    }

    /* Populate Array */
    private: static const int windowSize = 1925;
    private: float blackPixel[windowSize];

    public: void populateArr(string path, CascadeClassifier &eyeCascade) {
        cout << "Populating array..." << endl;
        int i = 0;
        VideoCapture cap(path);
        Mat frame;

        cap.read(frame);
        if (frame.empty()) return;
        Rect eyeRIO = detectEyes(frame, eyeCascade);

        while (true) {
            cap.read(frame);
            if (frame.empty() || i == windowSize) { break; }
            frame = frame(eyeRIO);
            Mat upper;
            formatFrame(frame, upper);
            
            int histSize = 256;
            float range[] = { 0, 256 }; //the upper boundary is exclusive
            const float* histRange[] = { range };
            bool uniform = true, accumulate = false;
            Mat hist;
            calcHist(&upper, 1, 0, Mat(), hist, 1, &histSize, histRange, uniform, accumulate);

            blackPixel[i] = hist.at<float>(0);
            i++;

            if (waitKey(1) == 27) {
                cout << "Program terminated." << endl;
                break;
            }
        }

        // for (int i = 0; i < windowSize; i++){
        //     cout << blackPixel[i] << " ";
        // }
        // cout << endl;
    }

    private: int counter = windowSize - 1;
    private: int close = 0, open = 0;
    private: int close_old = 0, open_old = 0;

    public: void detectBlink(Mat &frame) {
        Mat upper;
        formatFrame(frame, upper);
        
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
                // printf("[%d] blink\n", frameNo);
                blinkWindow = 20;
            }
        } else {
            blinkWindow -= 1;
        }

        imshow("upper", upper);
    }
};

class MyClass: public BlinkDetectionAlgorithm {
    public: void myMethod(string path, int ref[30], int code) {
        ContourDetectionAndAnalysis obj1;
        GlobalThresholdingOfNegative obj2;
        EyeBlackPixelRatioAnalysis obj3;
        QuickSortAndAnalysis obj4;

        CascadeClassifier eyeCascade;
        if (!eyeCascade.load("/home/pi/Desktop/opencv-test/haarcascade/haarcascade_eye_tree_eyeglasses.xml")) {
            std::cerr << "Could not load eye detector." << std::endl;
            return;
        }

        if (code == 4) {
            obj4.populateArr(path, eyeCascade);
        }

        VideoCapture cap(path);
        Mat frame;
        
        cap.read(frame);
        if (frame.empty()) return;
        Rect eyeRIO = detectEyes(frame, eyeCascade);

        while (true) {
            cap.read(frame);
            if (frame.empty()) break;
            frame = frame(eyeRIO);

            switch (code) {
                case 1: obj1.detectBlink(frame); break;
                case 2: obj2.detectBlink(frame); break;
                case 3: obj3.detectBlink(frame); break;
                default: obj4.detectBlink(frame); break;
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
            default: getBlinkAccuracy(obj4.detectedBlinks, ref); break;
        }
        cout << "test" << endl;
	}
};

int main() {
    int ref[9][30] = {
        {15,75,135,195,255,315,375,435,495,555,615,675,735,795,855,915,975,1035,1095,1155,1215,1275,1335,1395,1455,1515,1575,1635,1695,1755},{8,68,128,188,248,308,368,428,488,548,608,668,728,788,848,908,968,1028,1088,1148,1208,1268,1328,1388,1448,1508,1568,1628,1688,1748},{27,87,147,207,267,327,387,447,507,567,627,687,747,807,867,927,987,1047,1107,1167,1227,1287,1347,1407,1467,1527,1587,1647,1707,1767},{11,71,131,191,251,311,371,431,491,551,611,671,731,791,851,911,971,1031,1091,1151,1211,1271,1331,1391,1451,1511,1571,1631,1691,1751},{10,70,130,190,250,310,370,430,490,550,610,670,730,790,850,910,970,1030,1090,1150,1210,1270,1330,1390,1450,1510,1570,1630,1690,1750},{20,80,140,200,260,320,380,440,500,560,620,680,740,800,860,920,980,1040,1100,1160,1220,1280,1340,1400,1460,1520,1580,1640,1700,1760},{33,93,153,213,273,333,393,453,513,573,633,693,753,813,873,933,993,1053,1113,1173,1233,1293,1353,1413,1473,1533,1593,1653,1713,1773},{71,131,191,251,311,371,431,491,551,611,671,731,791,851,911,971,1031,1091,1151,1211,1271,1331,1391,1451,1511,1571,1631,1691,1751,1811},{27,87,147,207,267,327,387,447,507,567,627,687,747,807,867,927,987,1047,1107,1167,1227,1287,1347,1407,1467,1527,1587,1647,1707,1767}
    };
    
    string path[9] = {
        "/home/pi/Desktop/opencv-test/vid/0x 15y.mp4","/home/pi/Desktop/opencv-test/vid/0x -15y.mp4","/home/pi/Desktop/opencv-test/vid/0x -45y.mp4","/home/pi/Desktop/opencv-test/vid/75x 15y.mp4","/home/pi/Desktop/opencv-test/vid/75x -15y.mp4","/home/pi/Desktop/opencv-test/vid/75x -45y.mp4","/home/pi/Desktop/opencv-test/vid/-75x 15y.mp4","/home/pi/Desktop/opencv-test/vid/-75x -15y.mp4","/home/pi/Desktop/opencv-test/vid/-75x -45y.mp4"
    };

    MyClass contourDetectionAndAnalysis[9];
    // contourDetectionAndAnalysis[0].myMethod(path[0], ref[0], 4);
    for (int i = 0; i < 9; i++) {
        contourDetectionAndAnalysis[i].myMethod(path[i], ref[i], 1);
    }
    return 0;
}