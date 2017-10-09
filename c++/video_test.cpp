#include <stdio.h>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

int main(int argc, char** argv )
{
    VideoCapture cap("nvcamerasrc ! video/x-raw(memory:NVMM), 
width=(int)1280, height=(int)720,format=(string)I420, framerate=(fraction)24/1 ! nvvidconv flip-method=2 
! video/x-raw(memory:NVMM), format=(string)I420 ! nvoverlaysink -e ! appsink"); // open the camera
    if(!cap.isOpened()){ 
 // check if we succeeded
        return -1;
}

    Mat edges;
    namedWindow("edges",1);
    for(;;)
    {
        Mat frame;
        cap >> frame; // get a new frame from camera
        cvtColor(frame, edges, COLOR_BGR2GRAY);
        GaussianBlur(edges, edges, Size(7,7), 1.5, 1.5);
        Canny(edges, edges, 0, 30, 3);
        imshow("edges", edges);
       // if(waitKey(30) >= 0) break;
    }
    return 0;
}
