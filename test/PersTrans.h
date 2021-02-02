/*=========================================================
* 文 件 名：PersTrans.h
* 功能描述：透视变换算法头文件（包括主，前后，左右相机）
=========================================================*/

#ifndef PERSTRANS_H
#define PERSTRANS_H

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <cmath>
#include <iostream>

using namespace cv;
using namespace std;

Mat toushi_white(Mat image, Mat M, int border, int length, int width);
bool f_MainCam_PersTransMatCal(InputArray _src, int border_white, int border_biankuang, Mat *Mwhite, Mat *Mbiankuang, Mat *M_white_abshow, int ID, String ScreenType_Flag);
bool f_LeftRightCam_PersTransMatCal(InputArray _src, Mat *Mwhite, Mat *M_R_1_E, String ScreenType_Flag, int border_white);
bool f_FrontBackCam_PersTransMatCal(InputArray _src, Mat *Mwhite, String ScreenType_Flag);


#endif
