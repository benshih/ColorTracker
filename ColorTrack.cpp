/******************************************************
* Customized+modified by Benjamin Shih.
* Timeline: Sept 11 2014 - Sept 15 2014.
* Purpose: Position-based color tracker.
*
* Todo: 
* 9-15-14 needs compartmentalization.
* 
* Tutorial by Shermal Fernando as starter code:
* http://opencv-srf.blogspot.ch/2010/09/
* object-detection-using-color-seperation.html
*******************************************************/

#include <iostream>
#include <fstream>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

using namespace cv;
using namespace std;

int main(int argc, char** argv)
{
    /**** Change values here! ****/
    // Initial HSV values.
    int iLowH = 159;
    int iHighH = 179;

    int iLowS = 56;
    int iHighS = 255;
    
    int iLowV = 178;
    int iHighV = 248;

    const char* OUTFILENAME = "out.txt";


    // Parse input arguments. 
    VideoCapture inputVideo;
    if(2 != argc)
    {
        cout << "Usage: ./ColorTrack [video.filetype]" << endl;
        return -1;
    }
    else
    {
        inputVideo.open(string(argv[1]));
    }

    // Quit if invalid video file.
    if (!inputVideo.isOpened())
    {
        cout << "Cannot open file." << endl;
        return -1;
    }

    // Create separate windows to control
    // BSNote9-15-2014: there is some ordering issues even within the split
    // windows. I'm not sure why this happens.
    namedWindow("Hue", CV_WINDOW_AUTOSIZE);
    namedWindow("Saturation", CV_WINDOW_AUTOSIZE);
    namedWindow("Value", CV_WINDOW_AUTOSIZE);

    Size dimResize(800, 600);

    //Create trackbars in "Control" window
    createTrackbar("LowH", "Hue", &iLowH, 179); //Hue (0 - 179)
    createTrackbar("HighH", "Hue", &iHighH, 179);

    createTrackbar("LowS", "Saturation", &iLowS, 255); //Saturation (0 - 255)
    createTrackbar("HighS", "Saturation", &iHighS, 255);

    createTrackbar("LowV", "Value", &iLowV, 255);//Value (0 - 255)
    createTrackbar("HighV", "Value", &iHighV, 255);

    // Initial values for the tracked point.
    int iLastX = -1; 
    int iLastY = -1;

    // Capture a temporary image from the camerainputVideo
    Mat imgTmp;
    inputVideo.read(imgTmp); 

    // Create a black image with the size as the camera output
    Mat imgLines = Mat::zeros( imgTmp.size(), CV_8UC3 );;

/*
    // Configure output video parameters to match those of input video,
    // including codec.
    VideoWriter outputVideo;
    Size videoSize = Size((int) inputVideo.get(CV_CAP_PROP_FRAME_WIDTH), 
                          (int) inputVideo.get(CV_CAP_PROP_FRAME_HEIGHT));
    const string NAME = "output.mov";
    int ex = static_cast<int>(inputVideo.get(CV_CAP_PROP_FOURCC));
    outputVideo.open(NAME, ex, inputVideo.get(CV_CAP_PROP_FPS), videoSize, true);
    if(!outputVideo.isOpened())
    {
        cout << "Could not open output video file." << endl;
        return -1;
    }
*/

    // Initialize file stream for writing text to file.
    ofstream outfile;
    outfile.open(OUTFILENAME);

    // Cycle through the video's frames.
    bool readSuccess = true;
    while (readSuccess)
    {
        // Print out current HSV values to stdout.
        cout << "H: [" << iLowH << ", " << iHighH << "] S: [" << iLowS
            << ", " << iHighS << "] V: [" << iLowV << ", " << iHighV << "]" << endl;
        Mat imgOriginal;

        // Read a new frame from video.
        readSuccess = inputVideo.read(imgOriginal); // read a new frame from video

        Mat imgHSV;

        cvtColor(imgOriginal, imgHSV, COLOR_BGR2HSV); //Convert the captured frame from BGR to HSV

        Mat imgThresholded;

        inRange(imgHSV, Scalar(iLowH, iLowS, iLowV), Scalar(iHighH, iHighS, iHighV), imgThresholded); //Threshold the image

        //morphological opening (removes small objects from the foreground)
        erode(imgThresholded, imgThresholded, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)) );
        dilate( imgThresholded, imgThresholded, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)) ); 

        //morphological closing (removes small holes from the foreground)
        dilate( imgThresholded, imgThresholded, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)) ); 
        erode(imgThresholded, imgThresholded, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)) );

        //Calculate the moments of the thresholded image
        Moments oMoments = moments(imgThresholded);

        double dM01 = oMoments.m01;
        double dM10 = oMoments.m10;
        double dArea = oMoments.m00;

        // if the area <= 10000, I consider that the there are no object in the image and it's because of the noise, the area is not zero 
        if (dArea > 10000)
        {
            //calculate the position of the ball
            int posX = dM10 / dArea;
            int posY = dM01 / dArea;        

            if (iLastX >= 0 && iLastY >= 0 && posX >= 0 && posY >= 0)
            {
                //Draw a red line from the previous point to the current point
                line(imgLines, Point(posX, posY), Point(iLastX, iLastY), Scalar(0,0,255), 2);
            }

            iLastX = posX;
            iLastY = posY;
            outfile << posX << "," << posY << endl;
        }

        // Resize video file to something that fits on screens. Have not
        // tested this across multiple platforms.
        Mat imgOriginalResized;
        Mat imgThresholdedResized;
        
        resize(imgThresholded, imgThresholdedResized, dimResize, 0,
                0, INTER_CUBIC);

        imgOriginal = imgOriginal + imgLines;
        resize(imgOriginal, imgOriginalResized, dimResize, 0, 0,
                INTER_CUBIC);
        
        // Display the images.
        imshow("Thresholded Image", imgThresholdedResized); 
        imshow("Original", imgOriginalResized);

        // Write video to file.
        //outputVideo << imgOriginalResized;

        // Terminate program upon 'esc' key.
        if (27 == waitKey(30))
        {
            cout << "'esc' pressed to quit." << endl;
            outfile.close();
            return 1;
        }
    }
    cout << "No more frames in video." << endl;
    outfile.close();
    return 0;
}

/*
void closeFile(ofstream file)
{
    file.close();
}
*/
