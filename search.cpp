#include "stdafx.h"  
#include "fftw3.h"
#include "Header.h" 
#include <iostream>  
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

#define length 512

using namespace dlib;
using namespace cv;
using namespace std;

CvPoint SmoothROI(CvPoint2D32f newPt);

#define CHKRGN(pos) pos<0?0:pos

Mat img;

int main(int argc, const char** argv)
{
	try //如果沒找到輪廓不會當機
	{
		double fps;
		char string[10];
		double times = 0;

		int MRoiX = 0; //額頭Roi中心 (x, )
		int MRoiY = 0; //額頭Roi中心 ( ,y)

		int RRoiX = 0; //右臉Roi中心 (x, )
		int RRoiY = 0; //右臉Roi中心 ( ,y)

		int LRoiX = 0; //左臉Roi中心 (x, )
		int LRoiY = 0; //左臉Roi中心 ( ,y)

		CvPoint avgPtM;

		VideoCapture cap(0);

		int totalFrameNumber = cap.get(CV_CAP_PROP_FRAME_COUNT);
		cap.set(CV_CAP_PROP_FRAME_WIDTH, 640); //設定寬
		cap.set(CV_CAP_PROP_FRAME_HEIGHT, 480); //設定高

		// 載入學習檔
		frontal_face_detector detector = get_frontal_face_detector();
		shape_predictor pose_model;
		deserialize("shape_predictor_68_face_landmarks.dat") >> pose_model;

		bool _isInit = false;
//		Mat _Pos;

		Mat _PPos;
		Rect _Pos;

		bool paused = false;

		while (true)
		{
			times = (double)getTickCount();

				cap >> img;

				//Mat img_roi(320, 430, img.type());
				//int nChannels = img.channels();
				//int nRows = img.rows;
				//int nCols = img.cols;

				//for (int j = 60; j < nRows - 170; j++) {
				//	uchar* frameData = img.ptr<uchar>(j);
				//	uchar* roiData = img_roi.ptr<uchar>(j);
				//	for (int i = 220; i < nCols - 220; i++) {
				//		roiData[nChannels*i + 2] = frameData[nChannels*i + 2];
				//		roiData[nChannels*i + 1] = frameData[nChannels*i + 1];
				//		roiData[nChannels*i + 0] = frameData[nChannels*i + 0];
				//	}
				//}
				cv_image<bgr_pixel> cimg;

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

						int CenterX = (pt42.x() + pt39.x()) ; //印堂
						int CenterY = (pt42.y() + pt39.y()) ; //印堂

						MRoiX = CenterX - pt33.x();
						MRoiY = CenterY - pt33.y();

						//===========
						CvPoint2D32f newPtM = Point(MRoiX, MRoiY);
						avgPtM = SmoothROI(newPtM);

						if (avgPtM.x == 0 && avgPtM.y == 0) continue;

					}

					//------將 _Pos 放入人臉位置資訊 (左上x,y , 中心等)
//					_PPos = img(Rect(CHKRGN(avgPtM.x - 100), CHKRGN(avgPtM.y - 80), 200, 250));
					_Pos = Rect(CHKRGN(avgPtM.x - 100), CHKRGN(avgPtM.y - 80), 200, 250);

					_isInit = true;
				}
				else
				{
					//-------將 整張影像(img) 設定在ROI(_Pos+?????)的範圍
					//-------將設定好ROI的img copy 到 roiImg內

					//Mat test(_Pos.rows, _Pos.cols, img.type());
					//int nChannels = _Pos.channels();
					//int nRows = _Pos.rows;
					//int nCols = _Pos.cols;

					//for (int j = 0; j < nRows; j++) {
					//	uchar* srcData = _Pos.ptr<uchar>(j);
					//	uchar* dstData = test.ptr<uchar>(j);
					//	for (int i = 0; i < nCols; i++) {
					//		*dstData++ = *srcData++;
					//	}
					//}

					img = img(Rect(CHKRGN(avgPtM.x - 100)+_Pos.x, CHKRGN(avgPtM.y - 80)+_Pos.y, 200, 250));

					Mat roiImg;

					img.copyTo(roiImg);

					cimg = cv_image<bgr_pixel>(roiImg);

					// Detect faces 
					std::vector<dlib::rectangle> faces = detector(cimg);

					// Find the pose of each face.
					full_object_detection shape;
					for (unsigned short i = 0; i < faces.size(); ++i)
					{
						shape = pose_model(cimg, faces[i]);

						point pt39 = shape.part(39); //內眼角
						point pt42 = shape.part(42); //內眼角
						point pt33 = shape.part(33); //鼻頭

						//頭轉角度判斷
						point pt2 = shape.part(2); //右臉顴骨
						point pt36 = shape.part(36); //右外眼角
						point pt14 = shape.part(14); //左臉顴骨
						point pt45 = shape.part(45); //左外眼角
						point pt30 = shape.part(30); //鼻尖

						int CenterX = (pt42.x() + pt39.x()) ; //印堂
						int CenterY = (pt42.y() + pt39.y()) ; //印堂

						MRoiX = CenterX - pt33.x();
						MRoiY = CenterY - pt33.y();

						//===========
						CvPoint2D32f newPtM = Point(MRoiX, MRoiY);
						avgPtM = SmoothROI(newPtM);

						if (avgPtM.x == 0 && avgPtM.y == 0) continue;

						if (pt14.x() - pt30.x() <= 40)
						{
							CvPoint2D32f newPtR = Point(RRoiX, RRoiY);
							CvPoint avgPtR = SmoothROI(newPtR);
							Rect R_region_of_interest = Rect(CHKRGN(avgPtR.x - 5), CHKRGN(avgPtR.y - 5), 10, 10);
							cv::rectangle(img, R_region_of_interest, Scalar(0, 255, 0), 1, 8, 0);
						}
						else if (pt30.x() - pt2.x() <= 40)
						{
							CvPoint2D32f newPtL = Point(LRoiX, LRoiY + 70);
							CvPoint avgPtL = SmoothROI(newPtL);
							Rect L_region_of_interest = Rect(CHKRGN(avgPtL.x - 5), CHKRGN(avgPtL.y - 5), 10, 10);
							cv::rectangle(img, L_region_of_interest, Scalar(0, 255, 0), 1, 8, 0);
						}
						else
						{
							Rect region_of_interest = Rect(CHKRGN(avgPtM.x - 5), CHKRGN(avgPtM.y - 5), 10, 10);
							cv::rectangle(img, region_of_interest, Scalar(0, 255, 0), 1, 8, 0);
						}

						times = ((double)getTickCount() - times) / getTickFrequency();
						fps = 1.0 / times;
						sprintf(string, "%.2f", fps);
						std::string fpsString("FPS:");
						fpsString += string;
						putText(img, fpsString, Point(5, 20), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 255, 255));
					}
					//將 _Pos 放入人臉位置資訊 (左上x,y , 中心等)，但要記得再位置上做shift(人臉位置加上 _Pos.LeftTop Position)

					_Pos = Rect(CHKRGN(avgPtM.x - 100)+_Pos.x, CHKRGN(avgPtM.y - 80)+_Pos.y, 200, 250);


					//_Pos = img(Rect(CHKRGN(avgPtM.x - 100), CHKRGN(avgPtM.y - 80), 200, 250));
				}
					//imshow("1", img);
				
//				imshow("123", img_roi);

			imshow("Demo", img);

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
