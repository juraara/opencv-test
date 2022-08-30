#include "opencv2/opencv.hpp"
#include <iostream>

using namespace std;
using namespace cv;

int main(){

  /* Contour Detection */
	Mat frame, crop, gray, blur, thresh;
  Mat upper, lower; // upper and lower section of the eyes
	int minThresh = 70; // for thresholding
	int maxThresh = 255; // for thresholding

  // Create a VideoCapture object and use camera to capture the video
  VideoCapture cap(0); 

  // Check if camera opened successfully
  if(!cap.isOpened()){
   	cout << "Error opening video stream" << endl;
        return -1;
  }
  
  // Default resolutions of the frame are obtained.The default resolutions are system dependent.
  int frame_width = cap.get(cv::CAP_PROP_FRAME_WIDTH);
  int frame_height = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
  
  // Define the codec and create VideoWriter object.The output is stored in 'outcpp.avi' file.
  VideoWriter video("output-0.avi", cv::VideoWriter::fourcc('M','J','P','G'), 10, Size(frame_width,frame_height));

  char userInput;

  /* Pre-Recording */
  cout << "Press ESC key to start recording." << endl;
  while (1) {
    cap >> frame; // capture frame
    if (frame.empty()) break;
    
    crop = frame(Rect(170, 180, 230, 140)); // crop frame
		cvtColor(crop, gray, COLOR_BGR2GRAY); // convert to grayscale
		GaussianBlur(gray, blur, Size(9, 9), 0); // apply gaussian blur
		threshold(blur, thresh, minThresh, maxThresh, THRESH_BINARY_INV); // apply thresholding
		upper = thresh(Rect(0, 0, 230, 70)); // get upper eye
    lower = thresh(Rect(0, 70, 230, 70)); // get lower eye

    /* Display */   
    // imshow( "Frame", frame ); // display window
    imshow( "Crop", crop ); // display window
    line(gray, Point(0, 70), Point(230, 70), Scalar(0, 0, 255), 2, 8, 0);
    imshow("Grayscale", gray); // display window
    imshow("Threshold", thresh); // display window
    imshow("Upper", upper); // display window
    imshow("Lower", lower); // display window

    /* Press  ESC on keyboard to  exit */
    char d = (char)waitKey(1);
    if( d == 27 ) break;
  }

  /* Recording */
  cout << "Recording. Press ESC to stop recording." << endl;
  while (1) {
    cap >> frame; // capture frame
    if (frame.empty()) break;
    video.write(frame); // write to file
    
    crop = frame(Rect(170, 180, 230, 140)); // crop frame
		cvtColor(crop, gray, COLOR_BGR2GRAY); // convert to grayscale
		GaussianBlur(gray, blur, Size(9, 9), 0); // apply gaussian blur
		threshold(blur, thresh, minThresh, maxThresh, THRESH_BINARY_INV); // apply thresholding
		upper = thresh(Rect(0, 0, 230, 70)); // get upper eye
    lower = thresh(Rect(0, 70, 230, 70)); // get lower eye

    /* Display */   
    // imshow( "Frame", frame ); // display window
    line(gray, Point(0, 70), Point(230, 70), Scalar(0, 0, 255), 2, 8, 0);
    imshow("Grayscale", gray); // display window
    imshow("Threshold", thresh); // display window
    imshow("Upper", upper); // display window
    imshow("Lower", lower); // display window
 
    /* Press  ESC on keyboard to  exit */
    char c = (char)waitKey(1);
    if( c == 27 ) break;
  }

  cap.release();
  video.release();

  destroyAllWindows();
  return 0;
}
