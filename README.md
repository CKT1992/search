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


#define __VIDEO__POS

#if defined(__VIDEO__NORMAL)
#define VID_PATH "testforhr.mp4"
#elif defined(__VIDEO__POS)
#define VID_PATH "testforpose.mp4"
#elif defined(__VIDEO__SEARCH)
#define VID_PATH "testforsearch.mp4"
#elif defined(__VIDEO__CAMERA)
#define VID_PATH 0
#endif


#define LIMIT_SMOOTH_COUNT 7 //設定要smooth幾個frame
#define length 128

using namespace dlib;
using namespace cv;
using namespace std;

CvPoint SmoothROI(CvPoint2D32f newPt, bool isReInit = false);
CvPoint SmoothROI2(CvPoint2D32f newPt, bool isReInit = false);

#define CHKRGN(pos) pos<0?0:pos

float CalculateROIAverage(Mat, int);
float CalculateMOVEAverage();
//void SaveData(void);
void normalize();

float Origin_Average[300]; //原本長度
float MOVE_Average[300]; //移動平均
//float Standard[length];
float Detrended[length * 11], Detrended_r[length * 11]; //Origin - Average

int frameNumber = 0; //目前影格
int record = 0;

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

		VideoCapture cap(VID_PATH);
		if (cap.isOpened() == false)
		{
			cout << "Video load fail!" << endl;
		}
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
		Mat Roi_For_Cul;

		bool paused = false;

		bool _isReInit = false;
		// 0 ini
		// 1 front
		// 2 left
		// 3 right
		int  _type = 0;
		Point ROI_Center = Point(0, 0);
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
				if (faces.size() == 0) cout << "No faces detected!" << endl;
				for (unsigned short i = 0; i < faces.size(); ++i)
				{
					shape = pose_model(cimg, faces[i]);
					//SHAPE 代表找到的人臉位置資訊

					point pt39 = shape.part(39); //內眼角
					point pt42 = shape.part(42); //內眼角
					point pt33 = shape.part(33); //鼻頭

					int CenterX = (pt42.x() + pt39.x()); //印堂
					int CenterY = (pt42.y() + pt39.y()); //印堂

					MRoiX = CenterX - pt33.x();
					MRoiY = CenterY - pt33.y();

					//===========
					CvPoint2D32f newPtM = Point(MRoiX, MRoiY);
					avgPtM = SmoothROI(newPtM);
					imshow("Demo", img);
					if (avgPtM.x == 0 && avgPtM.y == 0) continue;

					
					ROI_Center = SmoothROI2(cvPoint2D32f(pt33.x(), pt42.y()));
					_Pos = Rect(0, 0, 640, 480);
					if (ROI_Center.x != 0 && ROI_Center.y != 0)
					{
						_isInit = true;
						_Pos = Rect(CHKRGN(ROI_Center.x - 250), CHKRGN(ROI_Center.y - 50), 500, 300);

					}


				}

				//------將 _Pos 放入人臉位置資訊 (左上x,y , 中心等)
//					_PPos = img(Rect(CHKRGN(avgPtM.x - 100), CHKRGN(avgPtM.y - 80), 200, 250));

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
				
				_PPos = img(_Pos);

				Mat roiImg = Mat(_Pos.height, _Pos.width, CV_8UC3);

				_PPos.copyTo(roiImg);

				cimg = cv_image<bgr_pixel>(roiImg);

				// Detect faces 
				std::vector<dlib::rectangle> faces = detector(cimg);

				// Find the pose of each face.
				full_object_detection shape;
				if (faces.size() == 0) cout << "No faces detected!" << endl;
				for (unsigned short i = 0; i < faces.size(); ++i)
				{
					shape = pose_model(cimg, faces[i]);

					point pt39 = shape.part(39); //右內眼角
					point pt42 = shape.part(42); //左內眼角
					point pt27 = shape.part(27); //鼻樑
					point pt33 = shape.part(33); //鼻頭

					//頭轉角度判斷
					point pt2 = shape.part(2); //右臉顴骨
					point pt36 = shape.part(36); //右外眼角
					point pt14 = shape.part(14); //左臉顴骨
					point pt45 = shape.part(45); //左外眼角
					point pt30 = shape.part(30); //鼻尖

					int CenterX = (pt42.x() + pt39.x()); //印堂
					int CenterY = (pt42.y() + pt39.y()); //印堂

					Point tArry[8];
					tArry[0] = Point(pt2.x(), pt2.y());
					tArry[1] = Point(pt39.x(), pt39.y());
					tArry[2] = Point(pt42.x(), pt42.y());
					tArry[3] = Point(pt33.x(), pt33.y());
					tArry[4] = Point(pt36.x(), pt36.y());
					tArry[5] = Point(pt14.x(), pt14.y());
					tArry[6] = Point(pt45.x(), pt45.y());
					tArry[7] = Point(pt30.x(), pt30.y());

					//for(int i = 0 ; i < 8 ; i++)
					cv::circle(_PPos, tArry[3], 3, Scalar(255, 0, 0));


					MRoiX = CenterX - pt33.x();
					MRoiY = CenterY - pt33.y();

					RRoiX = pt36.x();
					RRoiY = pt33.y();

					LRoiX = pt45.x();
					LRoiY = pt33.y();

					Rect region_of_interest;

					//===========

					CvPoint2D32f newPtM = Point(MRoiX + _Pos.x, MRoiY + _Pos.y);
					if (pt14.x() - pt30.x() <= 40)
					{
						if (_type != 3) {
							_isReInit = true;
							_type = 3;
						}
						CvPoint2D32f newPtR = Point(RRoiX + _Pos.x, RRoiY + _Pos.y);
						CvPoint avgPtR = SmoothROI(newPtR, _isReInit);
						region_of_interest = Rect(CHKRGN(avgPtR.x - 5), CHKRGN(avgPtR.y - 5), 10, 10); //拿來做事
						cv::rectangle(img, region_of_interest, Scalar(0, 255, 0), 1, 8, 0);
						_isReInit = false;
					}
					else if (pt30.x() - pt2.x() <= 40)
					{
						if (_type != 2) {
							_isReInit = true;
							_type = 2;
						}
						CvPoint2D32f newPtL = Point(LRoiX + _Pos.x, LRoiY + _Pos.y);
						CvPoint avgPtL = SmoothROI(newPtL, _isReInit);
						region_of_interest = Rect(CHKRGN(avgPtL.x - 5), CHKRGN(avgPtL.y - 5), 10, 10);
						cv::rectangle(img, region_of_interest, Scalar(0, 255, 0), 1, 8, 0);
						_isReInit = false;

					}
					else
					{
						if (_type != 1) {
							_isReInit = true;
							_type = 1;
						}
						avgPtM = SmoothROI(newPtM, _isReInit);
						region_of_interest = Rect(CHKRGN(avgPtM.x - 5), CHKRGN(avgPtM.y - 5), 10, 10);
						cv::rectangle(img, region_of_interest, Scalar(0, 255, 0), 1, 8, 0);
						_isReInit = false;
					}

					times = ((double)getTickCount() - times) / getTickFrequency();
					fps = 1.0 / times;
					sprintf(string, "%.2f", fps);
					std::string fpsString("FPS:");
					fpsString += string;
					cv::putText(img, fpsString, Point(5, 20), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 255, 255));
					ROI_Center = SmoothROI2(cvPoint2D32f(pt33.x(), pt42.y()));
					_Pos = Rect(CHKRGN(ROI_Center.x - 250), CHKRGN(ROI_Center.y - 50), 500, 300);

					//做事
					Roi_For_Cul = img(_Pos);
					Origin_Average[frameNumber] = CalculateROIAverage(Roi_For_Cul, 1);
					MOVE_Average[frameNumber] = CalculateMOVEAverage();
					frameNumber++;

				}

				imshow("Demo", img);
			}


			char c = (char)waitKey(10);
			if (c == 27)
				break;

			fix(c, paused);
		}
	}

	catch (exception& e)
	{
		cout << e.what() << endl;
		system("PAUSE");
	}
}

CvPoint SmoothROI(CvPoint2D32f newPt, bool isReInit)
{
	static std::queue<CvPoint2D32f> _queuePt;
	static CvPoint2D32f _sumPt = cvPoint2D32f(0.f, 0.f);
	if (isReInit)
	{
		while (_queuePt.size() > 0) _queuePt.pop();
		while(_queuePt.size() <LIMIT_SMOOTH_COUNT) _queuePt.push(newPt);
		_sumPt.x = newPt.x * (LIMIT_SMOOTH_COUNT);
		_sumPt.y = newPt.y * (LIMIT_SMOOTH_COUNT);
	}
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
CvPoint SmoothROI2(CvPoint2D32f newPt, bool isReInit)
{
	static std::queue<CvPoint2D32f> _queuePt;
	static CvPoint2D32f _sumPt = cvPoint2D32f(0.f, 0.f);
	if (isReInit)
	{
		while (_queuePt.size() > 0) _queuePt.pop();
		while (_queuePt.size() <LIMIT_SMOOTH_COUNT) _queuePt.push(newPt);
		_sumPt.x = newPt.x * (LIMIT_SMOOTH_COUNT);
		_sumPt.y = newPt.y * (LIMIT_SMOOTH_COUNT);
	}
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


float CalculateROIAverage(Mat roi, int channel)
{
	float avg = 0;

	if (roi.empty()) return 0;
	try
	{
		for (int x = 0; x < 10; x++)
		{
			for (int y = 0; y < 10; y++)
			{
				avg += (int)roi.at<Vec3b>(x, y)[channel];
			}
		}
		avg /= 100;
		return avg;
//		
//		for (int x = 0; x <= length; x++)
//		{
//			ROI_array[x] = avg;
//			cout << ROI_array[x];
//		}
//		for (int x = 0; x <= length + 2; x++)
//		{
//			if (x == 2)
//			{
//				MOVE_array[x] = (ROI_array[x-2] + ROI_array[x-1] + ROI_array[x] + ROI_array[x + 1] + ROI_array[x + 2])/5;
//			}
//		}
//		for (int x = 0; x <= length; x++)
//		{
//			OUT_array[x] = ROI_array[x] - MOVE_array[x];
//		}
//
//		
//
//		if (frameNumber % length == 0 && frameNumber > length) //256, 384, 512...
//		{
//			//moveaverage
//			for (int i = 0; i < length * 2; i++)
//			{
//				MoveAverage[frameNumber + i] = (Origin_g[frameNumber + i] + Origin_g[frameNumber + i + 1] + Origin_g[frameNumber + i + 2] + Origin_g[frameNumber + i + 3] + Origin_g[frameNumber + i + 4]) / 5;
//				Detrended[frameNumber + i] = Origin_g[frameNumber + i + 2] - MoveAverage[frameNumber + i];
//
////				normalize();
//
//				Detrended[frameNumber + i] = Origin_g[frameNumber + i + 2] - MoveAverage[frameNumber + i];
//				cout << Detrended[frameNumber + i];
//			}
//		}

	}
	catch (exception& e)
	{
		cout << "Calculate Fail" << endl;
	}
}

float CalculateMOVEAverage()
{
	static std::queue<float> _Move;
	static float _sumMove = 0;
	_Move.push(MOVE_Average[frameNumber]);
	_sumMove += _Move.front();
	if (_Move.size() <= LIMIT_SMOOTH_COUNT) return 0;
	float Mavg = 0;
	float firstPt = _Move.front();
	_sumMove -= firstPt;
	_Move.pop();
	Mavg = (float)(_sumMove / LIMIT_SMOOTH_COUNT);
	return Mavg;
}

void normalize(void)
{
	float average = 0, sd = 0;
	for (int i = 0; i < length * 2; i++)
		average += Detrended[i];
	average /= length * 2;

	for (int i = 0; i < length * 2; i++)
		sd += pow(Detrended[i] - average, 2);
	sd /= (length * 2 - 1);
	sd = sqrt(sd);

	for (int i = 0; i < length * 2; i++)
		Detrended[i] = (Detrended[i] - average) / sd;
}
