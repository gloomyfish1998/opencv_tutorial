#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <iostream>

using namespace cv;
using namespace cv::dnn;
using namespace std;

int main(int argc, char** argv) {
	string inference_pb = "D:/pedestrian_data/test/frozen_inference_graph.pb";
	string graph_txt = "D:/pedestrian_data/test/graph.pbtxt";
	Net net = readNetFromTensorflow(inference_pb, graph_txt);
	Mat image = imread("D:/python/Pedestrian-Detection/test_images/3600.jpg");
	int h = image.rows;
	int w = image.cols;
	imshow("input", image);

	Mat im_tensor = blobFromImage(image, 1.0, Size(300, 300), Scalar(), true, false);
	net.setInput(im_tensor);
	Mat cvOut = net.forward();
	Mat detectOut(cvOut.size[2], cvOut.size[3], CV_32F, cvOut.ptr<float>());
	for (int row = 0; row < detectOut.rows; row++) {
		float confidence = detectOut.at<float>(row, 2);
		if (confidence > 0.4) {
			int left = detectOut.at<float>(row, 3) * w;
			int top = detectOut.at<float>(row, 4) * h;
			int right = detectOut.at<float>(row, 5) * w;
			int bottom = detectOut.at<float>(row, 6) * h;

			Rect rect;
			rect.x = left;
			rect.y = top;
			rect.width = right - left;
			rect.height = bottom - top;
			rectangle(image, rect, Scalar(255, 0, 255), 2, 8, 0);
		}
	}
	imshow("detection out", image);
	waitKey(0);
	return 0;
}