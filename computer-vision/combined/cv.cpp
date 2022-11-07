#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <time.h>
#include <stdlib.h>

using namespace cv;
using namespace std;

class CV {
    
    public: 
        Mat rotate(Mat src, double angle)   //rotate function returning mat object with parametres imagefile and angle    
        {
            Mat dst;      //Mat object for output image file
            Point2f pt(src.cols/2., src.rows/2.);          //point from where to rotate    
            Mat r = getRotationMatrix2D(pt, angle, 1.0);      //Mat object for storing after rotation
            warpAffine(src, dst, r, Size(src.cols, src.rows));  ///applie an affine transforation to image.
            return dst;         //returning Mat object for output image file
        }

        void gammaCorrection(const Mat &src, Mat &dst, const float gamma) {
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

        Rect getLeftmostEye(vector<Rect> &eyes) {
            int leftmost = 99999999;
            int leftmostIndex = -1;
            for (int i = 0; i < eyes.size(); i++) {
                if (eyes[i].tl().x < leftmost) {
                    leftmost = eyes[i].tl().x;
                    leftmostIndex = i;
                }
            }
            return eyes[leftmostIndex];
        }

        Vec3f getEyeball(Mat &eye, vector<Vec3f> &circles)
            {
            vector<int> sums(circles.size(), 0);
            for (int y = 0; y < eye.rows; y++)
            {
                uchar *ptr = eye.ptr<uchar>(y);
                for (int x = 0; x < eye.cols; x++)
                {
                    int value = static_cast<int>(*ptr);
                    for (int i = 0; i < circles.size(); i++)
                    {
                        Point center((int)round(circles[i][0]), (int)round(circles[i][1]));
                        int radius = (int)round(circles[i][2]);
                        if (pow(x - center.x, 2) + pow(y - center.y, 2) < pow(eyeData.radius, 2))
                        {
                            sums[i] += value;
                        }
                    }
                    ++ptr;
                }
            }
            int smallestSum = 9999999;
            int smallestSumIndex = -1;
            for (int i = 0; i < circles.size(); i++)
            {
                if (sums[i] < smallestSum)
                {
                    smallestSum = sums[i];
                    smallestSumIndex = i;
                }
            }
            return circles[smallestSumIndex];
        }

        Point stabilize( vector<Point> &points, int windowSize)
        {
            float sumX = 0;
            float sumY = 0;
            int count = 0;
            for (int i = max(0, (int)(points.size() - windowSize)); i < points.size(); i++)
            {
                sumX += points[i].x;
                sumY += points[i].y;
                ++count;
            }
            if (count > 0)
            {
                sumX /= count;
                sumY /= count;
            }
            return Point(sumX, sumY);
        }

        struct EyeData {
            int radius = 0;
        };
        EyeData eyeData;

        Rect detectEyes(Mat &frame, CascadeClassifier &eyeCascade) {
            Mat gray;
            gammaCorrection(frame, frame, 5); // brighten image
            cvtColor(frame, gray, COLOR_BGR2GRAY); // convert image to grayscale
            equalizeHist(gray, gray); // enchance image contrast
            // Detect Both Eyes
            vector<Rect> eyes;
            eyeCascade.detectMultiScale(gray, eyes, 1.1, 2, 0 | CASCADE_SCALE_IMAGE, Size(90, 90)); // eye size (Size(90,90)) is determined emperically based on eye distance
            if (eyes.size() != 2) { // if both eyes not detected
                cout << "Error: Both eyes not detected" << endl;
                return Rect(0, 0, 0, 0); // return empty rectangle
            }
            for (Rect &eye : eyes) {
                rectangle(frame, eye.tl(), eye.br(), Scalar(0, 255, 0), 2); // draw rectangle around both eyes
            }
            // Get Left-Most Eyes and Detect Iris
            vector<Point> centers;
            vector<Vec3f> circles;
            Mat eye = gray(getLeftmostEye(eyes));
            HoughCircles(eye, circles, HOUGH_GRADIENT, 1, eye.cols / 8, 250, 15, eye.rows / 8, eye.rows / 3);
            if (circles.size() > 0) {
                Vec3f eyeball = getEyeball(eye, circles);
                Point center(eyeball[0], eyeball[1]);
                centers.push_back(center);
                center = stabilize(centers, 5); // we are using the last 5
                eyeData.radius = (int)eyeball[2]; // get iris radius
                circle(eye, center, eyeData.radius, Scalar(255, 255, 255), 2); // draw circle around the iris
            }
            else if (circles.size() <= 0) { // if eyeball not detected
                cout << "Error: Eyeball not detected" << endl;
                return Rect(0, 0, 0, 0); // return empty rectangle
            }

            line(eye, Point(0, getEyeball(eye, circles)[1]), Point(eye.cols, getEyeball(eye, circles)[1]), Scalar(255, 255, 255), 2, 8, 0);
            imshow("frame", frame);
            imshow("eye", eye);
            return getLeftmostEye(eyes);
        }

        float PERCLOS = 0.0;

        struct FrameData {
            static const int numOfFrames = 960; // num of frames = 1 minute @ avg 16fps
            float numOfPixels[numOfFrames] = { 0 }; // num of black/white pixels per frame
            int state[numOfFrames] = { 0 }; // frameData.state per frame
            int counter = 0;

            float prevFrameWhitePixelNo = 0.0;
            float currFrameWhitePixelNo = 0.0;
            bool close = false;
        };
        FrameData frameData;

        void detectBlink(Mat &frame) {
            // BGR to Binary
            Mat gray;
            cvtColor(frame, gray, COLOR_BGR2GRAY); // convert image to grayscale
            equalizeHist(gray, gray); // enchance image contrast
            Mat blur;
            GaussianBlur(gray, blur, Size(9, 9), 0); // blur image
            Mat thresh;
            threshold(blur, thresh, 20, 255, THRESH_BINARY_INV); // convert to binary image
            
            // Crop Sides to Remove Eyebrows etc.
            double crop_percent = 0.2;
            int x = thresh.cols * crop_percent;
            int y = thresh.rows * crop_percent;
            int src_w = thresh.cols * (1 - (crop_percent * 2));
            int src_h = thresh.rows * (1 - (crop_percent * 2));
            Mat crop = thresh(Rect(x, y, src_w, src_h)); // crop side to remove eyebrows etc.

            // Get Upper Half of Cropped Frame
            int upper_w = crop.cols;
            int upper_h = (int)((double)crop.rows * 0.50) + (int)((double)eyeData.radius * 0.3); // upper half and additional 3% of iris rad from the center should approximately include 80% of eyes.
            Mat upper = crop(Rect(0, 0, upper_w, upper_h)); // get upper half of image

            // Calculate Histogram
            int histSize = 256;
            float range[] = { 0, 256 }; // the upper boundary is exclusive
            const float* histRange[] = { range };
            bool uniform = true, accumulate = false;
            Mat hist;
            calcHist(&upper, 1, 0, Mat(), hist, 1, &histSize, histRange, uniform, accumulate); // get histogram
            
            // Compare Current and Previous Frames
            frameData.prevFrameWhitePixelNo = frameData.currFrameWhitePixelNo;
            frameData.currFrameWhitePixelNo = hist.at<float>(255);
            float percentDiff = ((frameData.prevFrameWhitePixelNo - frameData.currFrameWhitePixelNo) / ((frameData.prevFrameWhitePixelNo + frameData.currFrameWhitePixelNo) / 2)) * 100;
            if (percentDiff >= 80.0) {
                frameData.close = true;
            } 
            else if (percentDiff <= -20.0) {
                frameData.close = false;
            }
            
            // Calculate PERCLOS: P80
            frameData.numOfPixels[frameData.counter] = hist.at<float>(255);
            if (frameData.close) { 
                frameData.state[frameData.counter] = 1;
            }
            frameData.counter += 1;
            if (frameData.counter == frameData.numOfFrames) {
                // Get Number of Open and Closed States
                int closedStates = 0;
                for (int i = 0; i < frameData.numOfFrames; i++) {
                    closedStates += frameData.state[i];
                }

                // PERCLOS P:80
                PERCLOS = (float)closedStates / frameData.numOfFrames * 100;

                // SHL
                for (int i = 0; i < frameData.numOfFrames; i++) {
                    frameData.numOfPixels[i] = frameData.numOfPixels[i + 1];
                    frameData.state[i] = frameData.state[i + 1];
                }
                frameData.numOfPixels[frameData.numOfFrames - 1] = 0;
                frameData.state[frameData.numOfFrames - 1] = 0;
                frameData.counter = frameData.numOfFrames - 1;
            }

            Point p1(x, y);
            Point p2(x+src_h, y+src_w);
            rectangle(gray, p1, p2, Scalar(0, 255, 0), 2);
            line(gray, Point(x, y+upper_h), Point(x+src_w, y+upper_h), Scalar(0, 0, 255), 2, 8, 0);
            imshow("gray", gray);
            imshow("crop", crop);
            imshow("upper", upper);
        }
    
};

CV cV;

int main() {
    CascadeClassifier eyeCascade;
    if (!eyeCascade.load("haarcascade_eye_tree_eyeglasses.xml")) {
        cout << "xml not found! Check path and try again." << endl;
        return 0;
    }

    VideoCapture cap(0); // cap(0) (webcam), cap(1) (video)
    Mat frame;

    // Detect Eyes
    Rect eyeRIO = Rect(0, 0, 0, 0); // eye region of interest (ROI)
    while (eyeRIO.empty()) {
        cap.read(frame);
        cout << "Detecting eyes..." << endl; 
        if (frame.empty()) break;
        frame = cV.rotate(frame, 180);
        eyeRIO = cV.detectEyes(frame, eyeCascade);
    }

    // Detect Blinks
    while (true) {
        cap.read(frame);
        if (frame.empty()) break;
        if (eyeRIO.empty()) break;
        frame = cV.rotate(frame, 180);
        frame = frame(eyeRIO);
        cV.detectBlink(frame);
        cout << "PERCLOS = " << cV.PERCLOS << endl;
		
        if (waitKey(1) == 27) {
            cout << "Program terminated" << endl;
            break;
        }
    }

    return 0;
}

