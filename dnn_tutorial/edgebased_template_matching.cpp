#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

typedef struct PointInfo
{
 double Direction;
 double Magnitude;
 double MagnitudeN;
 double DerivativeX;
 double DerivativeY;
} ptin;

float minScore = 0.8;
float greediness = 0.8;
int main(int argc, char** argv) {
	Mat src = imread("D:/images/search2.jpg");
	Mat tpl = imread("D:/images/template.jpg");
	if (src.empty() || tpl.empty()) {
		printf("could not load images...\n");
		return -1;
	}

	namedWindow("source", WINDOW_AUTOSIZE);
	namedWindow("template", WINDOW_AUTOSIZE);
	imshow("source", src);
	imshow("template", tpl);

	Mat gray, binary;
	cvtColor(tpl, gray, COLOR_BGR2GRAY);
	Canny(gray, binary, 100, 800);
	vector<vector<Point>> contours;
	vector<Vec4i> hireachy;
	findContours(binary, contours, hireachy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE, Point(-1, -1));

	Mat gx, gy;
	Sobel(gray, gx, CV_32F, 1, 0);
	Sobel(gray, gy, CV_32F, 0, 1);

	Mat magnitude, direction;
	cartToPolar(gx, gy, magnitude, direction);
	long contoursLength = 0;
	double magnitudeTemp = 0;
	int originx = contours[0][0].x;
	int originy = contours[0][0].y;

	// 提取dx\dy\mag\log信息
	vector<vector<ptin>> contoursInfo;
	// 提取相对坐标位置
	vector<vector<Point>> contoursRelative;

	// 开始提取
	for (int i = 0; i < contours.size(); i++) {
		int n = contours[i].size();
		contoursLength += n;
		contoursInfo.push_back(vector<ptin>(n));
		vector<Point> points(n);
		for (int j = 0; j < n; j++) {
			int x = contours[i][j].x;
			int y = contours[i][j].y;
			points[j].x = x - originx;
			points[j].y = y - originy;
			ptin pointInfo;
			pointInfo.DerivativeX = gx.at<float>(y, x);
			pointInfo.DerivativeY = gy.at<float>(y, x);
			magnitudeTemp = magnitude.at<float>(y, x);
			pointInfo.Magnitude = magnitudeTemp;
			if (magnitudeTemp != 0)
				pointInfo.MagnitudeN = 1 / magnitudeTemp;
			contoursInfo[i][j] = pointInfo;
		}
		contoursRelative.push_back(points);
	}

	// 计算目标图像梯度
	Mat grayImage;
	cvtColor(src, grayImage, COLOR_BGR2GRAY);
	Mat gradx, grady;
	Sobel(grayImage, gradx, CV_32F, 1, 0);
	Sobel(grayImage, grady, CV_32F, 0, 1);

	Mat mag, angle;
	cartToPolar(gradx, grady, mag, angle);

	long totalLength = contoursLength;
	double nMinScore = minScore / totalLength; // normalized min score
	double nGreediness = (1 - greediness * minScore) / (1 - greediness) / totalLength;

	double partialScore = 0;
	double resultScore = 0;
	int resultX = 0;
	int resultY = 0;
	double start = (double)getTickCount();
	for (int row = 0; row < grayImage.rows; row++) {
		for (int col = 0; col < grayImage.cols; col++) {
			double sum = 0;
			long num = 0;
			for (int m = 0; m < contoursRelative.size(); m++) {
				for (int n = 0; n < contoursRelative[m].size(); n++) {
					num += 1;
					int curX = col + contoursRelative[m][n].x;
					int curY = row + contoursRelative[m][n].y;
					if (curX < 0 || curY < 0 || curX > grayImage.cols - 1 || curY > grayImage.rows - 1) {
						continue;
					}

					// 目标边缘梯度
					double sdx = gradx.at<float>(curY, curX);
					double sdy = grady.at<float>(curY, curX);

					// 模板边缘梯度
					double tdx = contoursInfo[m][n].DerivativeX;
					double tdy = contoursInfo[m][n].DerivativeY;

					// 计算匹配
					if ((sdy != 0 || sdx != 0) && (tdx != 0 || tdy != 0))
					{
						double nMagnitude = mag.at<float>(curY, curX);
						if (nMagnitude != 0)
							sum += (sdx * tdx + sdy * tdy) * contoursInfo[m][n].MagnitudeN / nMagnitude;
					}

					// 任意节点score之和必须大于最小阈值
					partialScore = sum / num;
					if (partialScore < min((minScore - 1) + (nGreediness * num), nMinScore * num))
						break;
				}
			}

			// 保存匹配起始点
			if (partialScore > resultScore)
			{
				resultScore = partialScore;
				resultX = col;
				resultY = row;
			}
		}
	}
	Point  offset_point(resultX, resultY);
	float time =(((double)getTickCount() - start)) / getTickFrequency();
	printf("edge template matching time : %.2f seconds\n", time);

	drawContours(src, contoursRelative, -1, Scalar(255, 0, 0), 2, 8, Mat(), INT_MAX, offset_point);
	imshow("edge-match", src);
	waitKey(0);
	return 0;
}