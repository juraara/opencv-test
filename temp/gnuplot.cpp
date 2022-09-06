#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>

#include "gnuplot-iostream.h"

int main(int argc, char* argv[])
{
    cv::Mat image = cv::imread(argv[1], CV_LOAD_IMAGE_COLOR);
    if (!image.data)
    {
        LOGMSG_ERR_S_C() << "No image data\n";
        return -1;
    }
    cv::cvtColor(image, image, cv::COLOR_BGR2GRAY);

    // construct a grayscale histogram
    int binNumber = 256;
    int channels[] = {0};
    int numImages = 1;
    int histDim = 1;
    int histSize[] = {binNumber};
    float grayRange[] = {0, 256};
    const float* ranges[] = {grayRange};
    cv::MatND hist;
    cv::calcHist(&image, numImages, channels, cv::Mat(), hist, histDim, histSize, ranges);

    // plot the historgram
    std::vector<std::pair<double, double> > data;
    for (int bin = 0; bin < binNumber; bin++)
    {
        data.emplace_back(bin, hist.at<float>(bin));
    }
    Gnuplot gp;
    gp << "plot" << gp.file1d(data) << "with lines title 'hist'" << std::endl;
    cv::imshow("gray", image);

    // show the image and wait for a keypress
    cv::waitKey(0);
    cv::destroyAllWindows();

    return 0;
}