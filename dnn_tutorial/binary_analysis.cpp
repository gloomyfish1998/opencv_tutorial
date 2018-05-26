#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;
vector<Rect> findBlocks(Mat &image, bool yellow);
vector<Rect> findWhitePoints(Mat &image, Mat binary);
bool inBlock(vector<Rect> &blocks, int cx, int cy);
int main(int argc, char** argv) {
	Mat src = imread("D:/javaopencv/daopian.png");
	namedWindow("input", CV_WINDOW_AUTOSIZE);
	imshow("input", src);

	// 颜色转换
	Mat hsv, mask1, mask2;
	cvtColor(src, hsv, COLOR_BGR2HSV);

	// 提取黄色与蓝色刀片在HSV色彩空间
	inRange(hsv, Scalar(26, 43, 46), Scalar(34, 255, 255), mask1); // 黄色
	inRange(hsv, Scalar(100, 43, 46), Scalar(124, 255, 255), mask2); // 蓝色

																	 // 获取黑色刀片位置
	vector<Rect> blackpieces = findWhitePoints(src, mask1);

	// 获取黄色刀片位置
	vector<Rect> yellowpieces = findBlocks(mask1, true);

	// 获取蓝色刀片位置
	vector<Rect> bluepieces = findBlocks(mask2, false);

	int bc = 0;
	for (size_t t = 0; t < blackpieces.size(); t++) {
		Rect r = blackpieces[t];
		int cx = r.x + r.width / 2;
		int cy = r.y + r.height / 2;
		if (inBlock(yellowpieces, cx, cy) || inBlock(bluepieces, cx, cy)) {
			continue;
		}
		bc++;
		rectangle(src, r, Scalar(255, 0, 255), 2, 8);
		circle(src, Point(cx, cy), 2, Scalar(255, 0, 255), -1);

	}

	// 绘制矩形与显示文字
	int total = yellowpieces.size() + bluepieces.size() + bc;
	printf("总刀片数量: %d\n", total);

	// 绘制黄色刀片
	for (size_t t = 0; t < yellowpieces.size(); t++) {
		Rect r = yellowpieces[t];
		rectangle(src, r, Scalar(0, 255, 255), 2, 8);
		circle(src, Point(r.x + r.width / 2, r.y + r.height / 2), 2, Scalar(0, 0, 255), -1);
	}
	// 绘制蓝色刀片
	for (size_t t = 0; t < bluepieces.size(); t++) {
		Rect r = bluepieces[t];
		rectangle(src, r, Scalar(255, 0, 0), 2, 8);
		circle(src, Point(r.x + r.width / 2, r.y + r.height / 2), 2, Scalar(0, 0, 255), -1);
	}
	putText(src, format("total: %d, yellow: %d, blue : %d, black: %d ", total, yellowpieces.size(), bluepieces.size(), bc),
		Point(10, 12), FONT_HERSHEY_PLAIN, 1.0, Scalar(0, 0, 255), 1, 8);

	imshow("result", src);
	waitKey(0);
	return 0;
}

vector<Rect> findBlocks(Mat &image, bool yellow) {
	vector<Rect> blocks;
	Mat k = yellow ? getStructuringElement(MORPH_RECT, Size(3, 3), Point(-1, -1)) :
		getStructuringElement(MORPH_RECT, Size(5, 5), Point(-1, -1));
	//erode(image, image, k);
	morphologyEx(image, image, MORPH_OPEN, k);
	vector<vector<Point>> contours;
	vector<Vec4i> hireachy;
	findContours(image, contours, hireachy, RETR_TREE, CHAIN_APPROX_SIMPLE);
	Mat result = Mat::zeros(image.size(), CV_8UC3);
	int count = 0;
	int hh = image.rows / 2;
	int ww = image.cols / 2;
	for (size_t t = 0; t < contours.size(); t++) {
		double area = contourArea(contours[t]);
		if (area < 400) continue;
		Rect rect = boundingRect(contours[t]);
		if (rect.height >= rect.width || rect.height > hh || rect.width > ww) continue;
		count++;
		drawContours(result, contours, t, Scalar(255, 0, 0), 2, 8);
		blocks.push_back(rect);
	}
	printf("number of blocks: %d\n", count);
	return blocks;
}

vector<Rect> findWhitePoints(Mat &image, Mat binary) {
	vector<Rect> blocks;
	Mat k = getStructuringElement(MORPH_RECT, Size(3, 3), Point(-1, -1));
	morphologyEx(binary, binary, MORPH_OPEN, k);
	vector<vector<Point>> contours;
	vector<Vec4i> hireachy;
	findContours(binary, contours, hireachy, RETR_TREE, CHAIN_APPROX_SIMPLE);
	int hh = binary.rows / 2;
	int ww = binary.cols / 2;
	int w = 0, h = 0;
	int x = 0, y = 0;
	for (size_t t = 0; t < contours.size(); t++) {
		double area = contourArea(contours[t]);
		if (area < 400) continue;
		Rect rect = boundingRect(contours[t]);
		if (rect.height >= rect.width || rect.height > hh || rect.width > ww) continue;
		// rectangle(image, rect, Scalar(0, 255, 255), 2, 8);

		// find max box for usage
		if (w < rect.width) {
			w = rect.width;
			x = rect.x;
		}
		if (h < rect.height) {
			h = rect.height;
			y = rect.y;
		}
		blocks.push_back(rect);
	}
	// try to find white
	Rect rr;
	rr.x = x;
	rr.y = y;
	rr.width = w;
	rr.height = image.rows - y * 2;

	Mat roi = image(rr);
	GaussianBlur(roi, roi, Size(3, 3), 0);
	Mat gray, mask;
	cvtColor(roi, gray, COLOR_BGR2HLS);
	threshold(gray, mask, 200, 255, THRESH_BINARY);
	cvtColor(mask, gray, COLOR_HLS2BGR);
	cvtColor(gray, gray, COLOR_BGR2GRAY);
	threshold(gray, mask, 200, 255, THRESH_BINARY);

	Mat k2 = getStructuringElement(MORPH_RECT, Size(5, 5), Point(-1, -1));
	dilate(mask, mask, k2);
	findContours(mask, contours, hireachy, RETR_TREE, CHAIN_APPROX_SIMPLE);
	int hw = mask.cols / 2;
	vector<Rect> blackpieces;
	for (size_t t = 0; t < contours.size(); t++) {
		double area = contourArea(contours[t]);
		if (area < 10) continue;
		Rect rect = boundingRect(contours[t]);
		if (rect.x > hw) continue;
		rect.width = w;
		rect.height = h;
		rect.x = x;
		rect.y = rect.y - h / 3;

		// offset plus
		rect.y = rr.y + rect.y;
		blackpieces.push_back(rect);
	}
	return blackpieces;
}

bool inBlock(vector<Rect> &blocks, int cx, int cy) {
	vector<Point> pts;
	bool inside = false;
	for (size_t t = 0; t < blocks.size(); t++) {
		Rect r = blocks[t];
		pts.push_back(Point(r.x, r.y));
		pts.push_back(Point(r.x + r.width, r.y));
		pts.push_back(Point(r.x + r.width, r.y + r.height));
		pts.push_back(Point(r.x, r.y + r.height));
		int dist = pointPolygonTest(pts, Point(cx, cy), false);
		pts.clear();
		printf("distance : %d\n", dist);
		if (dist > 0) {
			inside = true;
			break;
		}
	}
	return inside;
}