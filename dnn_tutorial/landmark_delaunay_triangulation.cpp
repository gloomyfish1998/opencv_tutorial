#include <opencv2/opencv.hpp>
#include <opencv2/face.hpp>
#include <iostream>

using namespace cv;
using namespace cv::face;
using namespace std;

RNG rng(12345);
String harr_file = "D:/opencv-3.4/opencv/build/etc/haarcascades/haarcascade_frontalface_alt_tree.xml";
int main(int argc, char** argv) {
	// Mat src = imread("D:/javaopencv/gaoyy.jpg");
	Mat src = imread("D:/vcprojects/images/persons.png");
	Mat result = src.clone();
	imshow("input", src);

	Mat gray;
	cvtColor(src, gray, COLOR_BGR2GRAY);
	equalizeHist(gray, gray);
	CascadeClassifier face_detector(harr_file);
	vector<Rect> faces;
	face_detector.detectMultiScale(gray, faces, 1.02, 1, 0, Size(20, 20), Size(300, 300));
	for (size_t t = 0; t < faces.size(); t++) {
		rectangle(src, faces[t], Scalar(0, 0, 255), 2, 8, 0);
	}
	imshow("face_detect", src);
	imwrite("D:/face_det.png", src);

	FacemarkLBF::Params params;
	params.n_landmarks = 68; // 68个标注点
	params.initShape_n = 10;
	params.stages_n = 5; // 算法的5个强化步骤
	params.tree_n = 6; // 模型中每个标注点结构树 数目
	params.tree_depth = 5; // 决策树深度

	// 创建LBF landmark 检测器
	Ptr<FacemarkLBF> facemark = FacemarkLBF::create(params);

	// 加载模型数据
	facemark->loadModel("D:/vcprojects/images/lbfmodel.yaml");
	cout << "Loaded model" << endl;

	// 提取人脸landmark-68个特征点
	vector<vector<Point2f>> landmarks;
	facemark->fit(src, faces, landmarks);

	Size size = src.size();
	Rect rect(0, 0, size.width, size.height);
	
	// 剖分三角形
	for (size_t t = 0; t < landmarks.size(); t++) {
		vector<Point2f> shapes = landmarks[t];

		// 创建剖分三角形生成器
		Subdiv2D subdiv;
		subdiv.initDelaunay(rect);

		// 添加与绘制特征点
		for (int i = 0; i < shapes.size(); i++) {
			subdiv.insert(shapes[i]);
			circle(result, shapes[i], 2, Scalar(0, 0, 255), -1, 8, 0);
		}
		
		// 生成剖分三角形
		vector<Vec6f> triangleList;
		subdiv.getTriangleList(triangleList);
		vector<Point> pt(3);

		// 绘制剖分三角形
		for (size_t i = 0; i < triangleList.size(); i++)
		{
			Vec6f t = triangleList[i];
			pt[0] = Point(cvRound(t[0]), cvRound(t[1]));
			pt[1] = Point(cvRound(t[2]), cvRound(t[3]));
			pt[2] = Point(cvRound(t[4]), cvRound(t[5]));

			// 使用随机颜色，绘制三角形
			if (rect.contains(pt[0]) && rect.contains(pt[1]) && rect.contains(pt[2]))
			{
				line(result, pt[0], pt[1], Scalar(rng.uniform(0, 255), rng.uniform(0, 256), rng.uniform(0, 256)), 1, LINE_AA, 0);
				line(result, pt[1], pt[2], Scalar(rng.uniform(0, 255), rng.uniform(0, 256), rng.uniform(0, 256)), 1, LINE_AA, 0);
				line(result, pt[2], pt[0], Scalar(rng.uniform(0, 255), rng.uniform(0, 256), rng.uniform(0, 256)), 1, LINE_AA, 0);
			}
		}
	}
	imshow("delaunay_triangle_demo", result);
	imwrite("D:/delaunay_result.png", result);
	waitKey(0);
	return 0;
}