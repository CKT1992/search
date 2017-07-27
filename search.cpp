#include "stdafx.h"  
#include "fftw3.h"
#include "Header.h"
#include <iostream>  
#include <conio.h>
#include <string>
#include <queue>
#include <fstream>  
#include <ctype.h>
#include <opencv2/opencv.hpp>  
#include <dlib/image_processing/frontal_face_detector.h>  
#include <dlib/image_processing/render_face_detections.h>  
#include <dlib/image_processing.h>
#include <dlib/gui_widgets.h>
#include <dlib/image_io.h>
#include <dlib/opencv.h>

#ifndef __SMOOTH_ROI__
#define __SMOOTH_ROI__

#define LIMIT_SMOOTH_COUNT 7 //設定要smooth幾個frame
#endif

using namespace dlib;
using namespace cv;
using namespace std;

CvPoint SmoothROI(CvPoint2D32f newPt);

#define CHKRGN(pos) pos<0?0:pos

Mat image, roiImage;
bool isFacedetected = false;

int main(int argc, const char** argv)
{
	try //如果沒找到輪廓不會當機
	{
		double fps;
		char string[10];
		double times = 0;

		int MRoiX = 0; //額頭Roi中心 (x, )
		int MRoiY = 0; //額頭Roi中心 ( ,y)

		VideoCapture cap(0);

		int totalFrameNumber = cap.get(CV_CAP_PROP_FRAME_COUNT);
		cap.set(CV_CAP_PROP_FRAME_WIDTH, 640); //設定寬
		cap.set(CV_CAP_PROP_FRAME_HEIGHT, 480); //設定高

		// 載入學習檔
		frontal_face_detector detector = get_frontal_face_detector();
		shape_predictor pose_model;
		deserialize("shape_predictor_68_face_landmarks.dat") >> pose_model;

		Mat frame;
		bool paused = false;

		for(;;)
		{
			times = (double)getTickCount();
			if (!paused)
			{
				cap >> frame;
				if (frame.empty())
					break;
			}

			frame.copyTo(image);

			if (!paused)
			{
				cv_image<bgr_pixel> cimage; //原本的640*480
				cv_image<bgr_pixel> cimage2;//搜尋框
				
				if (isFacedetected == false) 
				{
					cimage=image;
					isFacedetected = true;
				}
				else
				{
					cimage2=roiImage;
				}

				// Detect faces 
				std::vector<dlib::rectangle> faces = detector(cimage);
				std::vector<dlib::rectangle> faces2 = detector(cimage2);
				// Find the pose of each face.
				full_object_detection shape;
				for (unsigned short i = 0; i < faces.size(); ++i)
				{
					shape = pose_model(cimage, faces[i]);

					//設定感興趣空間
					point pt39 = shape.part(39); //內眼角
					point pt42 = shape.part(42); //內眼角
					point pt33 = shape.part(33); //鼻頭

					int CenterX = (pt42.x() + pt39.x()); //印堂
					int CenterY = (pt42.y() + pt39.y()); //印堂

					MRoiX = CenterX - pt33.x();
					MRoiY = CenterY - pt33.y();

					//===========
					CvPoint2D32f newPtM = Point(MRoiX, MRoiY);
					CvPoint avgPtM = SmoothROI(newPtM);

					if (avgPtM.x == 0 && avgPtM.y == 0) continue;

					Mat roi = image(Rect(CHKRGN(avgPtM.x - 100), CHKRGN(avgPtM.y - 80), 200, 250));


					Mat test;
					//int a = 0, b = 0;
					//for (int y = avgPtM.y - 100 ; y < avgPtM.y + 100; y++)
					//	for (int x = avgPtM.x - 80 ; x < avgPtM.x + 170 ; x++)
					//	{
					//		test.at<int>(a, b) = image.at<int>(x, y);
					//		if (a<200)
					//			a++;
					//		else
					//		{
					//			a = 0;
					//			b++;
					//		}
					//	}

					//顯示綠框
					Rect region_of_interest = Rect(CHKRGN(avgPtM.x - 5), CHKRGN(avgPtM.y - 5), 10, 10);
					cv::rectangle(image, region_of_interest, Scalar(0, 255, 0), 1, 8, 0);
					
					times = ((double)getTickCount() - times) / getTickFrequency();
					fps = 1.0 / times;
					sprintf(string, "%.1f", fps);
					std::string fpsString("FPS:");
					fpsString += string;
					putText(roi, fpsString, Point(5, 20), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 255, 255));

					roiImage = roi.clone();
					imshow("", roi);
				}

			}

			imshow("Demo", frame);

			char c = (char)waitKey(10);
			if (c == 27)
				break;

			fix(c, paused);
		}
	}

	catch (exception& e)
	{
		cout << e.what() << endl;
	}
}

CvPoint SmoothROI(CvPoint2D32f newPt)
{
	static std::queue<CvPoint2D32f> _queuePt;
	static CvPoint2D32f _sumPt = cvPoint2D32f(0.f, 0.f);
	_queuePt.push(newPt);
	_sumPt.x += newPt.x;
	_sumPt.y += newPt.y;
	if (_queuePt.size() <= LIMIT_SMOOTH_COUNT) return cvPoint(0, 0);
	CvPoint avgPt = cvPoint(0, 0);
	CvPoint2D32f firstPt = _queuePt.front();
	_sumPt.x -= firstPt.x;
	_sumPt.y -= firstPt.y;
	_queuePt.pop();
	avgPt.x = (int)(_sumPt.x / LIMIT_SMOOTH_COUNT);
	avgPt.y = (int)(_sumPt.y / LIMIT_SMOOTH_COUNT);
	return avgPt;
}
