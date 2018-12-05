#include <opencv2/opencv.hpp>
#include <opencv2/tracking.hpp>
#include <iostream>

using namespace cv;
using namespace std;

int main(int artc, char** argv) {
	Rect2d roi;
	Mat frame;

	// create a tracker object
	Ptr<Tracker> tracker = TrackerKCF::create();
	VideoCapture capture("D:/images/video/balltest.mp4");

	// select roi
	capture.read(frame);
	roi = selectROI("tracker", frame);
	// roi = selectROI(frame);

	// init tracker with roi
	tracker->init(frame, roi);

	// update tracking roi for each frame
	while (true) {
		bool ret = capture.read(frame);
		if (!ret) break;
		tracker->update(frame, roi);
		rectangle(frame, roi, Scalar(0, 0, 255), 2, 8, 0);
		imshow("tracker", frame);
		char c = waitKey(50);
		if (c == 27) {
			break;
		}
	}

	waitKey(0);
	return 0;
}
