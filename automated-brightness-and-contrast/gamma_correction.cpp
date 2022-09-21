
#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <vector>
#include <math.h>

using namespace cv;
using namespace std;

void gammaCorrection(const Mat &img, const double gamma_)
{
    CV_Assert(gamma_ >= 0);
    //! [changing-contrast-brightness-gamma-correction]
    Mat lookUpTable(1, 256, CV_8U);
    uchar* p = lookUpTable.ptr();
    for( int i = 0; i < 256; ++i)
        p[i] = saturate_cast<uchar>(pow(i / 255.0, gamma_) * 255.0);

    Mat res = img.clone();
    LUT(img, lookUpTable, res);
    //! [changing-contrast-brightness-gamma-correction]
    Mat img_gamma_corrected;
    hconcat(img, res, img_gamma_corrected);
    imshow("Gamma correction", img_gamma_corrected);
}

/* Main */
int main()
{
    Mat src = imread("eye-1.png");
    /* Convert to HSV */
    Mat hsv;
    cvtColor(src, hsv, COLOR_BGR2HSV);
    vector<Mat> hsv_planes;
    split( hsv, hsv_planes );
    
    /* Compute gamma */
    double mid = 1;
    cv::Scalar temp = cv::mean(hsv_planes[2]);
    float mean = temp.val[0];
    double gamma = log(mid*255)/log(mean);
    cout << "Gamma :" << gamma << endl;

    /* Do gamma correction on value channel */
    Mat dsrc;
    hsv_planes[2].convertTo(dsrc, CV_64F); // Convert to double for "pow"
    Mat ddst;
    pow(dsrc, gamma, ddst); // Compute the "pow"
    Mat val_gamma;
    ddst.convertTo(val_gamma, CV_8U); // Convert back to uchar

    /* Combine new value channel with original hue and sat channels */
    vector<Mat> channels;
    channels.push_back(hsv_planes[0]);
    channels.push_back(hsv_planes[1]);
    channels.push_back(val_gamma);
    Mat hsv_gamma;
    merge(channels, hsv_gamma);
    Mat img_gamma;
    cvtColor(hsv_gamma, img_gamma, COLOR_HSV2BGR);
    Mat gray1;
    cvtColor(img_gamma, gray1, COLOR_BGR2GRAY); // convert to grayscale
    equalizeHist(gray1, gray1);
    imshow("gray1", gray1);
    
    Mat gray2;
    cvtColor(src, gray2, COLOR_BGR2GRAY);
    equalizeHist(gray2, gray2);
    imshow("gray2", gray2);

    /* Show results */
    // imshow("input", src);
    // imshow("result", img_gamma);
    waitKey();

    waitKey(0);
    return 0;
}
