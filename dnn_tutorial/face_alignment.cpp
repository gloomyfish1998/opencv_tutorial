#include <opencv2/opencv.hpp>
#include <opencv2/face.hpp>
#include <math.h>
#include <iostream>

using namespace cv;
using namespace cv::face;
using namespace std;

const String  lbpfilePath = "D:/opencv-3.4/opencv/build/etc/lbpcascades/lbpcascade_frontalface.xml";
bool myDetector(InputArray image, OutputArray faces, CascadeClassifier *face_cascade);
void face_alignment(Mat &face, Point left, Point right, Rect roi);

int main(int argc, char** argv) {
	Mat img = imread("D:/vcprojects/images/gaoyy.png");
	namedWindow("input", CV_WINDOW_AUTOSIZE);
	imshow("input", img);

	CascadeClassifier face_cascade;
	face_cascade.load(lbpfilePath);

	FacemarkLBF::Params params;
	params.n_landmarks = 68; // 68个标注点
	params.initShape_n = 10;
	params.stages_n = 5; // 算法的5个强化步骤
	params.tree_n = 6; // 模型中每个标注点结构树 数目
	params.tree_depth = 5; // 决策树深度

	// 创建LBF landmark 检测器
	Ptr<FacemarkLBF> facemark = FacemarkLBF::create(params);
	facemark->setFaceDetector((FN_FaceDetector)myDetector, &face_cascade);

	// 加载模型数据
	facemark->loadModel("D:/vcprojects/images/lbfmodel.yaml");
	cout << "Loaded model" << endl;

	// 开始检测
	printf("start to detect landmarks...\n");
	vector<Rect> faces;
	facemark->getFaces(img, faces);
	vector< vector<Point2f> > shapes;
	if (facemark->fit(img, faces, shapes))
	{
		Point eye_left; // 36th
		Point eye_right; // 45th
		for (unsigned long i = 0; i<faces.size(); i++) {
			eye_left = shapes[i][36];
			eye_right = shapes[i][45];
			line(img, eye_left, eye_right, Scalar(255, 0, 0), 2, 8, 0);
			face_alignment(img(faces[i]), eye_left, eye_right, faces[i]);

			// 绘制人脸矩形区域
			rectangle(img, faces[i], Scalar(255, 0, 0));
			// 绘制人脸68个 landmark点位
			for (unsigned long k = 0; k<shapes[i].size(); k++)
				cv::circle(img, shapes[i][k], 2, cv::Scalar(0, 0, 255), FILLED);
		}
		namedWindow("Detected_shape");
		imshow("Detected_shape", img);
		waitKey(0);
	}
	return 0;
}

bool myDetector(InputArray image, OutputArray faces, CascadeClassifier *face_cascade)
{
	Mat gray;

	if (image.channels() > 1)
		cvtColor(image, gray, COLOR_BGR2GRAY);
	else
		gray = image.getMat().clone();

	equalizeHist(gray, gray);

	std::vector<Rect> faces_;
	face_cascade->detectMultiScale(gray, faces_, 1.1, 1, CASCADE_SCALE_IMAGE, Size(50, 50));
	Mat(faces_).copyTo(faces);
	return true;
}

void face_alignment(Mat &face, Point left, Point right, Rect roi) {
	int offsetx = roi.x;
	int offsety = roi.y;

	// 计算中心位置
	int cx = roi.width / 2;
	int cy = roi.height / 2;

	// 计算角度
	int dx = right.x - left.x;
	int dy = right.y - left.y;
	double degree = 180 * ((atan2(dy, dx)) / CV_PI);

	// 旋转矩阵计算
	Mat M = getRotationMatrix2D(Point2f(cx, cy), degree, 1.0);
	Point2f center(cx, cy);
	Rect bbox = RotatedRect(center, face.size(), degree).boundingRect();
	M.at<double>(0, 2) += (bbox.width / 2.0 - center.x);
	M.at<double>(1, 2) += (bbox.height / 2.0 - center.y);

	// 对齐
	Mat result;
	warpAffine(face, result, M, bbox.size());
	imshow("face-alignment", result);
}
