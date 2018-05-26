#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <iostream>

using namespace cv;
using namespace cv::xfeatures2d;
using namespace std;

void find_known_object(Mat &box, Mat &box_scene);
int main(int argc, char** argv) {

	Mat box = imread("D:/vcprojects/images/test_image/1/box.bmp");
	Mat scene = imread("D:/vcprojects/images//test_image/1/scene.jpg");
	imshow("box image", box);
	imshow("scene image", scene);
	find_known_object(box, scene);

	//Mat gray;
	//cvtColor(src, gray, COLOR_BGR2GRAY);
	/*int intHessian = 400;
	auto detector = SURF::create(intHessian);
	vector<KeyPoint> keypoints_box, keypoints_scene;
	Mat descriptor_box, descriptor_scene;
	detector->detectAndCompute(box, Mat(), keypoints_box, descriptor_box);
	detector->detectAndCompute(scene, Mat(), keypoints_scene, descriptor_scene);

	Ptr<FlannBasedMatcher> matcher = FlannBasedMatcher::create();
	vector<DMatch> matches;
	matcher->match(descriptor_box, descriptor_scene, matches);
	Mat dst;
	drawMatches(box, keypoints_box, scene, keypoints_scene, matches, dst);
	imshow("match-demo", dst);
	*/

	waitKey(0);
	return 0;
}

void find_known_object(Mat &box, Mat &box_scene) {
	int minHessian = 400;
	Ptr<SURF> detector = SURF::create();
	detector->setHessianThreshold(minHessian);
	std::vector<KeyPoint> keypoints_1, keypoints_2;
	Mat descriptors_1, descriptors_2;

	detector->detectAndCompute(box, Mat(), keypoints_1, descriptors_1);
	detector->detectAndCompute(box_scene, Mat(), keypoints_2, descriptors_2);

	// 计算匹配点
	FlannBasedMatcher matcher;
	std::vector< DMatch > matches;
	matcher.match(descriptors_1, descriptors_2, matches);
	double max_dist = 0; double min_dist = 100;

	// 计算最大与最小距离
	for (int i = 0; i < descriptors_1.rows; i++)
	{
		double dist = matches[i].distance;
		if (dist < min_dist) min_dist = dist;
		if (dist > max_dist) max_dist = dist;
	}
	printf("-- Max dist : %f \n", max_dist);
	printf("-- Min dist : %f \n", min_dist);

	// 寻找最佳匹配，距离越小越好
	std::vector< DMatch > good_matches;
	for (int i = 0; i < descriptors_1.rows; i++)
	{
		if (matches[i].distance <= max(3 * min_dist, 0.08))
		{
			good_matches.push_back(matches[i]);
		}
	}

	// 绘制最终匹配点
	Mat img_matches;
	drawMatches(box, keypoints_1, box_scene, keypoints_2,
		good_matches, img_matches, Scalar::all(-1), Scalar::all(-1),
		vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

	//-- Localize the object
	std::vector<Point2f> obj;
	std::vector<Point2f> scene;
	for (size_t i = 0; i < good_matches.size(); i++)
	{
		//-- Get the keypoints from the good matches
		obj.push_back(keypoints_1[good_matches[i].queryIdx].pt);
		scene.push_back(keypoints_2[good_matches[i].trainIdx].pt);
	}
	Mat H = findHomography(obj, scene, RHO);
	// 无法配准
	// Mat H = findHomography(obj, scene, RANSAC);

	//-- Get the corners from the image_1 ( the object to be "detected" )
	std::vector<Point2f> obj_corners(4);
	obj_corners[0] = cvPoint(0, 0); obj_corners[1] = cvPoint(box.cols, 0);
	obj_corners[2] = cvPoint(box.cols, box.rows); obj_corners[3] = cvPoint(0, box.rows);
	std::vector<Point2f> scene_corners(4);
	perspectiveTransform(obj_corners, scene_corners, H);

		//-- Draw lines between the corners (the mapped object in the scene - image_2 )
	line(img_matches, scene_corners[0] + Point2f(box.cols, 0), scene_corners[1] + Point2f(box.cols, 0), Scalar(0, 255, 0), 4);
	line(img_matches, scene_corners[1] + Point2f(box.cols, 0), scene_corners[2] + Point2f(box.cols, 0), Scalar(0, 255, 0), 4);
	line(img_matches, scene_corners[2] + Point2f(box.cols, 0), scene_corners[3] + Point2f(box.cols, 0), Scalar(0, 255, 0), 4);
	line(img_matches, scene_corners[3] + Point2f(box.cols, 0), scene_corners[0] + Point2f(box.cols, 0), Scalar(0, 255, 0), 4);

	//-- Show detected matches
	imshow("Good Matches & Object detection", img_matches);
	imwrite("D:/result.png", img_matches);
	waitKey(0);
}