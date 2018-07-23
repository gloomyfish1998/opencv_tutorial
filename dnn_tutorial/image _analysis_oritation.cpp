/************************************************************************/
//《炼数成金》课程使用
// by jsxyhelu
/************************************************************************/
#include "stdafx.h"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/photo.hpp"
 

using namespace std;
using namespace cv;

//获得构建的主要方向，在图上进行标徽，并且返回角度结果
double getOrientation(vector<Point> &pts, Mat &img)
{
	//构建pca数据。这里做的是将轮廓点的x和y作为两个维压到data_pts中去。
	Mat data_pts = Mat(pts.size(), 2, CV_64FC1);//使用mat来保存数据，也是为了后面pca处理需要
	for (int i = 0; i < data_pts.rows; ++i)
	{
		data_pts.at<double>(i, 0) = pts[i].x;
		data_pts.at<double>(i, 1) = pts[i].y;
	}
	//执行PCA分析
	PCA pca_analysis(data_pts, Mat(), CV_PCA_DATA_AS_ROW);
	//获得最主要分量，在本例中，对应的就是轮廓中点，也是图像中点
	Point pos = Point(pca_analysis.mean.at<double>(0, 0),pca_analysis.mean.at<double>(0, 1));
	//存储特征向量和特征值
	vector<Point2d> eigen_vecs(2);
	vector<double> eigen_val(2);
	for (int i = 0; i < 2; ++i)
	{
		eigen_vecs[i] = Point2d(pca_analysis.eigenvectors.at<double>(i, 0),pca_analysis.eigenvectors.at<double>(i, 1));
		eigen_val[i] = pca_analysis.eigenvalues.at<double>(i,0);//注意，这个地方原代码写错了
	}
	//在轮廓/图像中点绘制小圆
	circle(img, pos, 3, CV_RGB(255, 0, 255), 2);
	//计算出直线，在主要方向上绘制直线
	line(img, pos, pos + 0.02 * Point(eigen_vecs[0].x * eigen_val[0], eigen_vecs[0].y * eigen_val[0]) , CV_RGB(255, 255, 0));
	line(img, pos, pos + 0.02 * Point(eigen_vecs[1].x * eigen_val[1], eigen_vecs[1].y * eigen_val[1]) , CV_RGB(0, 255, 255));
	//返回角度结果
	return atan2(eigen_vecs[0].y, eigen_vecs[0].x);
}

//程序主要部分
int main( int argc, char** argv )
{
	//读入图像，转换为灰度
	Mat img = imread("e:/sandbox/pca1.jpg");
	Mat bw;
	double dRet;
	cvtColor(img, bw, COLOR_BGR2GRAY);
	//阈值处理
	threshold(bw, bw, 150, 255, CV_THRESH_BINARY);
	//寻找轮廓
	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;
	findContours(bw, contours, hierarchy, CV_RETR_LIST, CV_CHAIN_APPROX_NONE);
	//轮廓分析，找到工件
	for (size_t i = 0; i < contours.size(); ++i)
	{
		//计算轮廓大小
		double area = contourArea(contours[i]);
		//去除过小或者过大的轮廓区域（科学计数法表示）
		if (area < 1e2 || 1e5 < area) continue;
		//绘制轮廓
		drawContours(img, contours, i, CV_RGB(255, 0, 0), 2, 8, hierarchy, 0);
		//寻找每一个轮廓的方向
		dRet = getOrientation(contours[i], img);
	}
	printf("当前角度为%f",dRet);
	waitKey();
	return 0;
}
