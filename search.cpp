#include "stdafx.h"  
#include <iostream>  
#include <opencv2/opencv.hpp>  
#include <dlib/image_processing/frontal_face_detector.h>  
#include <dlib/image_processing/render_face_detections.h>  
#include <dlib/image_processing.h>
#include <dlib/gui_widgets.h>
#include <dlib/image_io.h>
#include <dlib/opencv.h>

using namespace dlib;
using namespace cv;
using namespace std;

#ifndef __SMOOTH_ROI__
#define __SMOOTH_ROI__

#define LIMIT_SMOOTH_COUNT 7 //設定要smooth幾個frame
#endif

CvPoint SmoothROI(CvPoint2D32f newPt);
#define CHKRGN(pos) pos<0?0:pos

int main()
{
	try //如果沒找到輪廓不會當機
	{
		double fps;
		char string[10];
		double time = 0;

		int MRoiX = 0; //額頭Roi中心 (x, )
		int MRoiY = 0; //額頭Roi中心 ( ,y)
		CvPoint avgPtM;

		Mat img;
		VideoCapture cap(0);
		image_window win;

		// 載入學習檔
		frontal_face_detector detector = get_frontal_face_detector();
		shape_predictor pose_model;
		deserialize("shape_predictor_68_face_landmarks.dat") >> pose_model;

		bool _isInit = false;
		Rect _Pos;

		for (;;)
		{
			time = (double)getTickCount();
			if (waitKey(50) == 30) { break; }

			cap >> img;

			cv_image<bgr_pixel> cimg;


			//判斷第一次人臉位置(預設只找到一個人臉)
			if (_isInit == false)
			{
				//給定整張影像
				cimg = cv_image<bgr_pixel>(img);

				//偵測人臉
				std::vector<dlib::rectangle> faces = detector(cimg);

				// Find the pose of each face.
				full_object_detection shape;
				for (unsigned short i = 0; i < faces.size(); ++i)
				{
					shape = pose_model(cimg, faces[i]);
					//SHAPE 代表找到的人臉位置資訊

					point pt39 = shape.part(39); //內眼角
					point pt42 = shape.part(42); //內眼角
					point pt33 = shape.part(33); //鼻頭

					int CenterX = (pt42.x() + pt39.x())/2; //印堂
					int CenterY = (pt42.y() + pt39.y())/2; //印堂

					MRoiX = CenterX - pt33.x();
					MRoiY = CenterY - pt33.y();

					//===========
					CvPoint2D32f newPtM = Point(MRoiX, MRoiY);
					avgPtM = SmoothROI(newPtM);

					if (avgPtM.x == 0 && avgPtM.y == 0) continue;
					
				}

				//------將 _Pos 放入人臉位置資訊 (左上x,y , 中心等)
				_Pos = Rect(CHKRGN(avgPtM.x - 100), CHKRGN(avgPtM.y - 80), 200, 250);


				_isInit = true;
			}
			else
			{
				//-------將 整張影像(img) 設定在ROI(_Pos+?????)的範圍
				//-------將設定好ROI的img copy 到 roiImg內

				Mat roiImg(_Pos.height, _Pos.width, CV_8UC3);


				img.copyTo(roiImg);

				cimg = cv_image<bgr_pixel>(roiImg);

				// Detect faces 
				std::vector<dlib::rectangle> faces = detector(cimg);

				// Find the pose of each face.
				full_object_detection shape;
				for (unsigned short i = 0; i < faces.size(); ++i)
				{
					shape = pose_model(cimg, faces[i]);

					time = ((double)getTickCount() - time) / getTickFrequency();
					fps = 1.0 / time;
					sprintf(string, "%.2f", fps);
					std::string fpsString("FPS:");
					fpsString += string;
					putText(img, fpsString, Point(5, 20), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 255, 255));
				}
				//將 _Pos 放入人臉位置資訊 (左上x,y , 中心等)，但要記得再位置上做shift(人臉位置加上 _Pos.LeftTop Position)
			}
			

			win.clear_overlay(); //畫面清空
			win.set_image(cimg); //顯示
			win.set_title("LIVE"); //視窗名稱
//			win.add_overlay(render_face_detections(shape)); //畫輪廓
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
