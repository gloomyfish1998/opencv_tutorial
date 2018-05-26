#include <opencv2/opencv.hpp>
#include <iostream>
#define PI 3.1415926

using namespace cv;
using namespace std;

RNG rng(12345);
void find_Lanes(Mat &frame);
int main() {
	String win_title = "input frame";
	namedWindow(win_title, CV_WINDOW_AUTOSIZE);
	VideoCapture capture("D:/case_four/videos/lane.avi");
	if (!capture.isOpened()) {
		printf("could not load video file");
		return -1;
	}
	Mat frame;
	while (capture.read(frame)) {
		imshow(win_title, frame);
		find_Lanes(frame);
		char c = waitKey(10);
		if (c == 27) {
			break;
		}
	}

	waitKey(0);
	return 0;
}

void find_Lanes(Mat &frame) {
	int offx = frame.cols/5;
	int offy = frame.rows / 3;
	Rect rect;
	rect.x = offx;
	rect.y = frame.rows- offy;
	rect.width = frame.cols- offx*2;
	rect.height = offy - 50;

	Mat copy = frame(rect).clone();
	GaussianBlur(copy, copy, Size(3, 3), 0);
	Mat gray, binary;
	cvtColor(copy, gray, COLOR_BGR2GRAY);
	threshold(gray, binary, 0, 255, THRESH_BINARY | THRESH_OTSU);
	Mat mask = Mat::zeros(frame.size(), CV_8UC1);
	binary.copyTo(mask(rect));

	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;
	findContours(mask, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE, Point(0, 0));
	Mat drawing = Mat::zeros(mask.size(), CV_8UC3);
	for (size_t i = 0; i< contours.size(); i++)
	{
		RotatedRect rrt = minAreaRect(contours[i]);
		int angle = abs(rrt.angle);
		if (angle < 20 || angle > 160 || angle == 90)
			continue;
		printf("rrt.angle: %.2f\n", rrt.angle);
		Scalar color = Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
		//drawContours(frame, contours, (int)i, color, 2, 8, hierarchy, 0, Point());
		Point pt1(-1, -1);
		Point pt2(-1, -1);
		int miny = 100000;
		int maxy = 0;
		for (int p = 0; p < contours[i].size(); p++) {
			Point onep = contours[i][p];
			if (miny > onep.y) {
				miny = onep.y;
				pt1.y = onep.y;
				pt1.x = onep.x;
			}
			if (maxy < onep.y) {
				maxy = onep.y;
				pt2.y = onep.y;
				pt2.x = onep.x;
			}
		}
		if (pt1.x < 0 || pt2.x< 0)
			continue;
		printf("line Point1 (x = %d, y = %d) to Point2 (x=%d, y=%d)\n", pt1.x, pt1.y, pt2.x, pt2.y);
		line(frame, pt1, pt2, Scalar(255, 0, 0), 3, 8);
	}

	imshow("mask", mask);
	imshow("lane-lines", frame);
}