#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>

#include <fstream>
#include <iostream>
#include <algorithm>
#include <cstdlib>
using namespace std;
using namespace cv;
using namespace cv::dnn;

float confidenceThreshold = 0.25;
void video_detection();
void image_detection();
int main(int argc, char** argv)
{
	image_detection();
}

void video_detection() {
	String modelConfiguration = "D:/vcprojects/images/dnn/yolov2-tiny-voc/yolov2-tiny-voc.cfg";
	String modelBinary = "D:/vcprojects/images/dnn/yolov2-tiny-voc/yolov2-tiny-voc.weights";
	dnn::Net net = readNetFromDarknet(modelConfiguration, modelBinary);
	if (net.empty())
	{
		printf("Could not load net...\n");
		return;
	}

	vector<string> classNamesVec;
	ifstream classNamesFile("D:/vcprojects/images/dnn/yolov2-tiny-voc/voc.names");
	if (classNamesFile.is_open())
	{
		string className = "";
		while (std::getline(classNamesFile, className))
			classNamesVec.push_back(className);
	}

	// VideoCapture capture(0); 
	VideoCapture capture;
	capture.open("D:/vcprojects/images/fbb.avi");
	if (!capture.isOpened()) {
		printf("could not open the camera...\n");
		return;
	}

	Mat frame;
	while (capture.read(frame))
	{
		if (frame.empty())
			if (frame.channels() == 4)
				cvtColor(frame, frame, COLOR_BGRA2BGR);
		Mat inputBlob = blobFromImage(frame, 1 / 255.F, Size(416, 416), Scalar(), true, false);
		net.setInput(inputBlob, "data");
		Mat detectionMat = net.forward("detection_out");
		vector<double> layersTimings;
		double freq = getTickFrequency() / 1000;
		double time = net.getPerfProfile(layersTimings) / freq;
		ostringstream ss;
		ss << "FPS: " << 1000 / time << " ; time: " << time << " ms";
		putText(frame, ss.str(), Point(20, 20), 0, 0.5, Scalar(0, 0, 255));

		for (int i = 0; i < detectionMat.rows; i++)
		{
			const int probability_index = 5;
			const int probability_size = detectionMat.cols - probability_index;
			float *prob_array_ptr = &detectionMat.at<float>(i, probability_index);
			size_t objectClass = max_element(prob_array_ptr, prob_array_ptr + probability_size) - prob_array_ptr;
			float confidence = detectionMat.at<float>(i, (int)objectClass + probability_index);
			if (confidence > confidenceThreshold)
			{
				float x = detectionMat.at<float>(i, 0);
				float y = detectionMat.at<float>(i, 1);
				float width = detectionMat.at<float>(i, 2);
				float height = detectionMat.at<float>(i, 3);
				int xLeftBottom = static_cast<int>((x - width / 2) * frame.cols);
				int yLeftBottom = static_cast<int>((y - height / 2) * frame.rows);
				int xRightTop = static_cast<int>((x + width / 2) * frame.cols);
				int yRightTop = static_cast<int>((y + height / 2) * frame.rows);
				Rect object(xLeftBottom, yLeftBottom,
					xRightTop - xLeftBottom,
					yRightTop - yLeftBottom);
				rectangle(frame, object, Scalar(0, 255, 0));
				if (objectClass < classNamesVec.size())
				{
					ss.str("");
					ss << confidence;
					String conf(ss.str());
					String label = String(classNamesVec[objectClass]) + ": " + conf;
					int baseLine = 0;
					Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
					rectangle(frame, Rect(Point(xLeftBottom, yLeftBottom),
						Size(labelSize.width, labelSize.height + baseLine)),
						Scalar(255, 255, 255), CV_FILLED);
					putText(frame, label, Point(xLeftBottom, yLeftBottom + labelSize.height),
						FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 0));
				}
			}
		}
		imshow("YOLOv3: Detections", frame);
		if (waitKey(1) >= 0) break;
	}
}

void image_detection() {
	String modelConfiguration = "D:/vcprojects/images/dnn/yolov2-tiny-voc/yolov2-tiny-voc.cfg";
	String modelBinary = "D:/vcprojects/images/dnn/yolov2-tiny-voc/yolov2-tiny-voc.weights";
	dnn::Net net = readNetFromDarknet(modelConfiguration, modelBinary);
	if (net.empty())
	{
		printf("Could not load net...\n");
		return;
	}
	vector<string> classNamesVec;
	ifstream classNamesFile("D:/vcprojects/images/dnn/yolov2-tiny-voc/voc.names");
	if (classNamesFile.is_open())
	{
		string className = "";
		while (std::getline(classNamesFile, className))
			classNamesVec.push_back(className);
	}

	// 加载图像
	Mat frame = imread("D:/vcprojects/images/fastrcnn.jpg");
	Mat inputBlob = blobFromImage(frame, 1 / 255.F, Size(416, 416), Scalar(), true, false);
	net.setInput(inputBlob, "data");

	// 检测
	Mat detectionMat = net.forward("detection_out");
	vector<double> layersTimings;
	double freq = getTickFrequency() / 1000;
	double time = net.getPerfProfile(layersTimings) / freq;
	ostringstream ss;
	ss << "detection time: " << time << " ms";
	putText(frame, ss.str(), Point(20, 20), 0, 0.5, Scalar(0, 0, 255));

	// 输出结果
	for (int i = 0; i < detectionMat.rows; i++)
	{
		const int probability_index = 5;
		const int probability_size = detectionMat.cols - probability_index;
		float *prob_array_ptr = &detectionMat.at<float>(i, probability_index);
		size_t objectClass = max_element(prob_array_ptr, prob_array_ptr + probability_size) - prob_array_ptr;
		float confidence = detectionMat.at<float>(i, (int)objectClass + probability_index);
		if (confidence > confidenceThreshold)
		{
			float x = detectionMat.at<float>(i, 0);
			float y = detectionMat.at<float>(i, 1);
			float width = detectionMat.at<float>(i, 2);
			float height = detectionMat.at<float>(i, 3);
			int xLeftBottom = static_cast<int>((x - width / 2) * frame.cols);
			int yLeftBottom = static_cast<int>((y - height / 2) * frame.rows);
			int xRightTop = static_cast<int>((x + width / 2) * frame.cols);
			int yRightTop = static_cast<int>((y + height / 2) * frame.rows);
			Rect object(xLeftBottom, yLeftBottom,
				xRightTop - xLeftBottom,
				yRightTop - yLeftBottom);
			rectangle(frame, object, Scalar(0, 0, 255), 2, 8);
			if (objectClass < classNamesVec.size())
			{
				ss.str("");
				ss << confidence;
				String conf(ss.str());
				String label = String(classNamesVec[objectClass]) + ": " + conf;
				int baseLine = 0;
				Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
				rectangle(frame, Rect(Point(xLeftBottom, yLeftBottom),
					Size(labelSize.width, labelSize.height + baseLine)),
					Scalar(255, 255, 255), CV_FILLED);
				putText(frame, label, Point(xLeftBottom, yLeftBottom + labelSize.height),
					FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 0));
			}
		}
	}
	imshow("YOLO-Detections", frame);
	waitKey(0);
	return;
}