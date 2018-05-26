#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

String haar_data_file = "D:/opencv-3.4/opencv/build/etc/haarcascades/haarcascade_frontalface_alt_tree.xml";
String lbp_data_file = "D:/opencv-3.4/opencv/build/etc/lbpcascades/lbpcascade_frontalface_improved.xml";

void faceDemo();

class CascadeDetectorAdapter : public DetectionBasedTracker::IDetector
{
public:
	CascadeDetectorAdapter(Ptr<CascadeClassifier> detector) :
		IDetector(),
		Detector(detector)
	{
		CV_Assert(detector);
	}

	void detect(const cv::Mat &Image, std::vector<cv::Rect> &objects)
	{
		Detector->detectMultiScale(Image, objects, scaleFactor, 1, 0, Size(100, 100), Size(400, 400));
	}

	virtual ~CascadeDetectorAdapter()
	{}

private:
	CascadeDetectorAdapter();
	Ptr<CascadeClassifier> Detector;
};

int main(int argc, char** argv) {
	/*Mat src = imread("D:/vcprojects/images/greenback.png");
	VideoCapture capture;
	capture.open("D:/vcprojects/images/sample.mp4");
	namedWindow("input image", CV_WINDOW_AUTOSIZE);
	namedWindow("result", CV_WINDOW_AUTOSIZE);
	CascadeClassifier face_detector(haar_data_file);
	vector<Rect> faces;
	Mat gray;
	while (capture.read(src)) {
		imshow("input image", src);
		cvtColor(src, gray, COLOR_BGR2GRAY);
		float time = getTickCount();
		face_detector.detectMultiScale(gray, faces, 1.1, 1, 0, Size(20, 20), Size(150, 150));
		float end = (getTickCount() - time)/getTickFrequency();
		//printf("time consume : %.2f", end * 1000);

		for (size_t t = 0; t < faces.size(); t++) {
			rectangle(src, faces[t], Scalar(0, 0, 255), 2, 8, 0);
		}
		char c = waitKey(10);
		if (c == 27) {
			break;
		}
		imshow("result", src);
	}
	*/
	faceDemo();
	waitKey(0);
	return 0;
}

void faceDemo() {
	Mat frame, gray;
	VideoCapture capture;
	capture.open("D:/vcprojects/images/facerecog.mp4");
	namedWindow("input", CV_WINDOW_AUTOSIZE);
	String cascadeFrontalfilename = "D:/opencv-3.4/opencv/build/etc/haarcascades/haarcascade_frontalface_alt_tree.xml";
	DetectionBasedTracker::Parameters params;

	Ptr<CascadeClassifier> cascade = makePtr<cv::CascadeClassifier>(cascadeFrontalfilename);
	Ptr<DetectionBasedTracker::IDetector> MainDetector = makePtr<CascadeDetectorAdapter>(cascade);

	cascade = makePtr<cv::CascadeClassifier>(cascadeFrontalfilename);
	Ptr<DetectionBasedTracker::IDetector> TrackingDetector = makePtr<CascadeDetectorAdapter>(cascade);

	DetectionBasedTracker tracker(MainDetector, TrackingDetector, params);
	vector<Rect> faces;
	while (capture.read(frame)) {
		cvtColor(frame, gray, COLOR_RGB2GRAY);
		tracker.process(gray);
		tracker.getObjects(faces);
		if (faces.size()) {
			for (size_t i = 0; i < faces.size(); i++)
			{
				rectangle(frame, faces[i], Scalar(0, 255, 0));
			}
		}
		imshow("input", frame);
		char c = waitKey(10);
		if (c == 27) {
			break;
		}
	}
	tracker.stop();

}