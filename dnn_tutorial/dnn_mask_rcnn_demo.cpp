#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <iostream>

using namespace cv;
using namespace cv::dnn;
using namespace std;

const size_t width = 300;
const size_t height = 300;

//String pb_model = "D:/tensorflow/mask_rcnn_inception_v2_coco_2018_01_28/frozen_inference_graph.pb";
//String pb_txt = "D:/tensorflow/mask_rcnn_inception_v2_coco_2018_01_28/graph.pbtxt";

String pb_model = "E:/MarkRcnn/MaskRcnn_bp/MaskRcnn.pb";
String pb_txt = "E:/MarkRcnn/MaskRcnn_bp/MaskRcnn.pbtxt";
int main(int argc, char** argv) {
	Mat frame = imread("E:/MarkRcnn/testImages/20190228085410.jpg");
	// Mat frame = imread("D:/images/zhigang.png");
	if (frame.empty()) {
		printf("could not load image...\n");
		return -1;
	}
	namedWindow("input image", WINDOW_AUTOSIZE);
	imshow("input image", frame);

	// 加载模型
	Net net = readNetFromTensorflow(pb_model, pb_txt);
	net.setPreferableBackend(DNN_BACKEND_OPENCV);
	net.setPreferableTarget(DNN_TARGET_CPU);

	// 设置数据
	Mat blobImage = blobFromImage(frame, 1.0, Size(800, 800), Scalar(0, 0, 0), true, false);
	printf("blobImage width : %d, height: %d\n", blobImage.size[3], blobImage.size[2]);
	net.setInput(blobImage);

	// 推理
	vector<String> out_names;
	out_names.push_back("detection_out_final");
	out_names.push_back("detection_masks");
	vector<Mat> outs;
	net.forward(outs, out_names);
	Mat detection = outs[0];

	int id = outs[1].size[0];
	int numClasses = outs[1].size[1];
	int mh = outs[1].size[2];
	int mw = outs[1].size[3];
	Mat masks = outs[1]; // Nx90x15x15
	printf("id: %d, numClasses:%d, m:%d, s:%d \n", id, numClasses, mh, mw);

	// 解析对象检测输出
	Mat detectionMat(detection.size[2], detection.size[3], CV_32F, detection.ptr<float>());
	float confidence_threshold = 0.6;
	for (int i = 0; i < detectionMat.rows; i++) {
		float confidence = detectionMat.at<float>(i, 2);
		if (confidence > confidence_threshold) {
			size_t objIndex = (size_t)(detectionMat.at<float>(i, 1));
			float tl_x = detectionMat.at<float>(i, 3) * frame.cols;
			float tl_y = detectionMat.at<float>(i, 4) * frame.rows;
			float br_x = detectionMat.at<float>(i, 5) * frame.cols;
			float br_y = detectionMat.at<float>(i, 6) * frame.rows;

			Rect object_box((int)tl_x, (int)tl_y, (int)(br_x - tl_x), (int)(br_y - tl_y));
			rectangle(frame, object_box, Scalar(255, 0, 255), 1, 8, 0);
			putText(frame, format(" confidence %.2f", confidence), Point(tl_x - 10, tl_y - 5), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(255, 0, 0), 2, 8);
			
			// 解析mask
			Mat mask(masks.size[2], masks.size[3], CV_32F, masks.ptr<float>(i, objIndex));
			Mat color_mask = Mat::zeros(mask.size(), CV_8UC3);
			Mat bin_mask = Mat::zeros(mask.size(), CV_8UC1);
			for (int row = 0; row < color_mask.rows; row++) {
				for (int col = 0; col < color_mask.cols; col++) {
					float m = mask.at<float>(row, col);
					if (m >= 0.5) {
						color_mask.at<Vec3b>(row, col) = Vec3b(0, 0, 255);
						bin_mask.at<uchar>(row, col) = 255;
					}
				}
			}
			Mat roi = frame(object_box);
			resize(color_mask, color_mask, roi.size());
			resize(bin_mask, bin_mask, roi.size());
			Mat result;
			bitwise_and(roi, roi, result, bin_mask);
			imshow("mask", result);
			addWeighted(roi, 0.5, color_mask, 0.5, 0, roi);
		}
	}
	imshow("mask-rcnn-demo", frame);

	waitKey(0);
	return 0;
}