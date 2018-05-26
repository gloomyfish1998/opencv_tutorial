#include <opencv2/opencv.hpp>
#include <opencv2/tracking.hpp>
#include <iostream>
/**
* 绿幕贴图 代码演示
* @auther zhigang jia
* @date 2018-05-11
* opencv3.4 + vs2015 + 64bit
*/
using namespace cv;
using namespace std;

void blendMask(Mat &src, Mat &dst, Mat &mask);
int main(int argc, char** argv) {
	Mat src1 = imread("D:/javaopencv/green.jpg");
	Mat src2 = imread("D:/javaopencv/xiaomaolu.jpg");
	if (src1.empty() || src2.empty()) {
		printf("could not load image..\n");
		return -1;
	}
	Mat hsv, mask;
	cvtColor(src1, hsv, COLOR_BGR2HSV);
	//imshow("hsv", hsv);
	inRange(hsv, Scalar(35, 43, 46), Scalar(77, 255, 255), mask);
	//imshow("mask", mask);

	Mat k = getStructuringElement(MORPH_RECT, Size(3, 3), Point(-1, -1));
	morphologyEx(mask, mask, MORPH_OPEN, k);
	
	resize(src1, src1, src2.size());
	resize(mask, mask, src2.size());

	blendMask(src1, src2, mask);
	waitKey(0);
	return 0;
}

void blendMask(Mat &src, Mat &dst, Mat &mask) {
	Mat blur_mask, blur_mask_f;

	GaussianBlur(mask, blur_mask, Size(3, 3), 0.0);
	blur_mask.convertTo(blur_mask_f, CV_32F);
	normalize(blur_mask_f, blur_mask_f, 1.0, 0.0, NORM_MINMAX);

	int w = src.cols;
	int h = src.rows;
	int ch = src.channels();
	for (int row = 0; row < h; row++) {
		for (int col = 0; col < w; col++) {
			int b1 = src.at<Vec3b>(row, col)[0];
			int g1 = src.at<Vec3b>(row, col)[1];
			int r1 = src.at<Vec3b>(row, col)[2];

			int b2 = dst.at<Vec3b>(row, col)[0];
			int g2 = dst.at<Vec3b>(row, col)[1];
			int r2 = dst.at<Vec3b>(row, col)[2];

			float w2 = blur_mask_f.at<float>(row, col);
			float w1 = 1 - w2;

			b2 = (int)(b2*w2 + b1*w1);
			g2 = (int)(g2*w2 + g1*w1);
			r2 = (int)(r2*w2 + r1*w1);

			dst.at<Vec3b>(row, col)[0] = b2;
			dst.at<Vec3b>(row, col)[1] = g2;
			dst.at<Vec3b>(row, col)[2] = r2;
		}
	}
	imwrite("D:/result.png", dst);
	imshow("blend mask", dst);
}
