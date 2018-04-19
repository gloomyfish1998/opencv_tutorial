#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <iostream>
/******************************************************
×
* 作者：贾志刚
* QQ: 57558865
* OpenCV DNN 完整视频教程：
× http://edu.51cto.com/course/11516.html
*
********************************************************/
using namespace cv;
using namespace cv::dnn;
using namespace std;
String labels_txt_file = "D:/vcprojects/images/dnn/synset_words.txt";
String model_bin_file = "D:/vcprojects/VGGNet/VGG_ILSVRC_16_layers.caffemodel";
String model_txt_file = "D:/vcprojects/VGGNet/VGG_ILSVRC_16_layers_deploy.prototxt";
vector<String> readLabels();
int main(int argc, char** argv) {
	Mat src = imread("D:/vcprojects/images/twocat.png");
	if (src.empty()) {
		printf("could not load image...\n");
		return -1;
	}
	namedWindow("input", CV_WINDOW_AUTOSIZE);
	imshow("input", src);
	vector<String> labels = readLabels();

	Mat rgb;
	cvtColor(src, rgb, COLOR_BGR2RGB);
	int w = 224;
	int h = 224;

	Net net = readNetFromCaffe(model_txt_file, model_bin_file);
	if (net.empty()) {
		printf("read caffe model data failure...\n");
		return -1;
	}
	Mat inputBlob = blobFromImage(src, 1.0, Size(w, h), Scalar(104, 117, 123));
	Mat prob;
	for (int i = 0; i < 10; i++) {
		net.setInput(inputBlob, "data");
		prob = net.forward("prob");
	}
	Mat probMat = prob.reshape(1, 1);
	Point classNumber;
	double classProb;
	minMaxLoc(probMat, NULL, &classProb, NULL, &classNumber);
	int classidx = classNumber.x;
	printf("\n current image classification : %s, possible : %.2f", labels.at(classidx).c_str(), classProb);

	putText(src, labels.at(classidx), Point(20, 20), FONT_HERSHEY_SIMPLEX, 1.0, Scalar(0, 0, 255), 2, 8);
	imshow("Image Classification", src);
	imwrite("D:/result.png", src);
	waitKey(0);
	return 0;
}

vector<String> readLabels() {
	vector<String> classNames;
	ifstream fp(labels_txt_file);
	if (!fp.is_open()) {
		printf("could not open the file");
		exit(-1);
	}
	string name;
	while (!fp.eof()) {
		getline(fp, name);
		if (name.length()) {
			classNames.push_back(name.substr(name.find(' ') + 1));
		}
	}
	fp.close();
	return classNames;
}
