#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <cmath>
#include <iostream>
#include <time.h>
#include <string.h>
#include "PersTrans.h"
#include "time.h"//统计时间需添加的头文件

using namespace cv;
using namespace std;

/*
与异物误检率漏检率密切相关的有关参数：

漏检率最低：
1.第一轮最小面积限定 3
2.第一轮最大面积限定 500
3.gabor图的内外围最小灰度差（第一轮判定） 6.1
4.gabor图的内外围最小灰度差（第二轮判定） 5.4
5.原图图的内外围最小灰度差（第二轮判定） 8.8
6.自适应阈值二值化的输入偏移常量     5.5

误检率最低：
1.第一轮最小面积限定 12  
2.第一轮最大面积限定 400
3.gabor图的内外围最小灰度差（第一轮判定） 7.7
4.gabor图的内外围最小灰度差（第二轮判定） 7.0
5.原图图的内外围最小灰度差（第二轮判定） 10.0
6.自适应阈值二值化的输入偏移常量     5.5

漏检率最低（刘海屏）：
1.第一轮最小面积限定 5
2.第一轮最大面积限定 400
3.gabor图的内外围最小灰度差（第一轮判定） 5.9
4.gabor图的内外围最小灰度差（第二轮判定） 5.2
5.原图图的内外围最小灰度差（第二轮判定） 8.6
6.自适应阈值二值化的输入偏移常量     5.5

说明：参数6越大，二值化分割出来的白色区域面积越小，数量越少，该数值大小建议在5.5-7.5之间
*/

bool Shifting(Mat white,Mat *mresult, String *causecolor,int num);
void adaptiveThresholdCustom(const cv::Mat &src, cv::Mat &dst, double maxValue, int method, int type, int blockSize, double delta, double ratio);
Mat Gabor7(Mat img_1);
void adaptiveThresholdCustom(const cv::Mat &src, cv::Mat &dst, double maxValue, int method, int type, int blockSize, double delta, double ratio);


#define yiwu_pre_size           51
#define yiwu_pre_th             5.5


void main()
{
	Mat White_Main, White_Main1;
	Mat Mwhite, Mblack, Mlouguang, Mabshow;
	Mat SideLight_Main, SideLight_Main1;
	bool result = false;
	String causeColor_1_white;
	Mat Mresult_1_white;

	Mat src_L1 = imread("G:\\背光源样本\\20210111样本\\移位样本\\23\\06_20200927165_23_112.bmp", -1);
	Mat src_R1 = imread("G:\\背光源样本\\20210111样本\\移位样本\\23\\06_20200927165_23_012.bmp", -1);
	Mat src_ceguang_left = imread("G:\\背光源样本\\20210115样本\\移位误检\\53\\10_20200927165_53_110.bmp", -1);
	Mat src_ceguang_right = imread("G:\\背光源样本\\20210115样本\\移位误检\\53\\10_20200927165_53_010.bmp", -1);

	Mat M_L_1, M_R_1, M_L_1_E, M_R_1_E;
	if (src_L1.channels() == 3)
		cvtColor(src_L1, src_L1, CV_BGR2GRAY);
	if (src_R1.channels() == 3)
		cvtColor(src_R1, src_R1, CV_BGR2GRAY);
	if (src_ceguang_left.channels() == 3)
		cvtColor(src_ceguang_left, src_ceguang_left, CV_BGR2GRAY);
	if (src_ceguang_right.channels() == 3)
		cvtColor(src_ceguang_right, src_ceguang_right, CV_BGR2GRAY);
	//主黑白相机处理
	bool Ext_Result_Left = f_LeftRightCam_PersTransMatCal(src_L1, &M_L_1, &M_L_1_E, "R角水滴屏", 15);
	bool Ext_Result_Right = f_LeftRightCam_PersTransMatCal(src_R1, &M_R_1, &M_R_1_E, "R角水滴屏", 15);

	Mat ceL1 = toushi_white(src_L1, M_L_1, -5, 3000, 1500);
	Mat ceR1 = toushi_white(src_R1, M_R_1, -5, 3000, 1500);
	Mat LeftCeGuang = toushi_white(src_ceguang_left, M_L_1, -5, 3000, 1500);      //左相机侧光校正图
	Mat RightCeGuang = toushi_white(src_ceguang_right, M_R_1, -5, 3000, 1500);    //右相机侧光校正图

	Mat src_L1_gray, src_R1_gray;
	src_L1_gray = src_L1.clone();
	src_R1_gray = src_R1.clone();

	Mat th2, th3;
	threshold(src_L1_gray, th2, 20, 255, CV_THRESH_BINARY);
	threshold(src_R1_gray, th3, 20, 255, CV_THRESH_BINARY);
	Mat left_mask = toushi_white(th2, M_L_1, -1, 3000, 1500);
	Mat right_mask = toushi_white(th3, M_R_1, -1, 3000, 1500);
	bitwise_and(left_mask, ceL1, ceL1);
	bitwise_and(right_mask, ceR1, ceR1);
	bitwise_and(left_mask, LeftCeGuang, LeftCeGuang);
	bitwise_and(right_mask, RightCeGuang, RightCeGuang);

	//Mat LeftCeGuang_enlarge = toushi_white(src_ceguang_left, M_L_1_E, -5, 3000, 1500);
	//Mat RightCeGuang_enlarge = toushi_white(src_ceguang_right, M_R_1_E, -5, 3000, 1500);
	//gabor滤波
	Mat leftfilter = Gabor7(ceL1);       //左侧白底滤波
	Mat rightfilter = Gabor7(ceR1);     //右侧白底滤波

	result = Shifting(leftfilter, &Mresult_1_white, &causeColor_1_white,1);
	if (!result)
	{
		result = Shifting(rightfilter, &Mresult_1_white, &causeColor_1_white,0);
	}

	cout << result << endl;
}

//比较函数对象
bool compareContourAreas(std::vector< cv::Point> contour1,std::vector< cv::Point> contour2)
{
	return (cv::contourArea(contour1) > cv::contourArea(contour2));
}

/*====================================================================
* 函 数 名: Shifting
* 功能描述:移位，表现为白底图象有一条亮线
* 输入：主相机白底图像
* 输出：主相机白底下检测结果图和result
* 其他：
======================================================================*/
bool Shifting(Mat white, Mat *mresult, String *causecolor,int num)
{
	bool result = false;
	Mat img_gray = white.clone();

	Mat ad_result, th1, th2, ImageBinary;
	//adaptiveThreshold(img_gray, ad_result, 255, CV_ADAPTIVE_THRESH_MEAN_C, CV_THRESH_BINARY_INV, 15, 3);

	adaptiveThresholdCustom(img_gray, th1, 255, ADAPTIVE_THRESH_MEAN_C, CV_THRESH_BINARY_INV, 51, 5.5, 1);

	Mat th_result = Mat::zeros(th1.rows - 2 * 200, th1.cols - 2 * 300, img_gray.type());
	th_result.copyTo(th1(Rect(300, 200, th1.cols - 2 * 300, th1.rows - 2 * 200)));
	vector<vector<Point>> contours;

	threshold(img_gray, ImageBinary, 30, 255, CV_THRESH_BINARY);
	bitwise_and(ImageBinary, th1, th2);

	//th1(Rect(0, 0, th1.cols - 1, 4)) = uchar(0);
	//th1(Rect(0, th1.rows-4, th1.cols - 1, 4)) = uchar(0);
	//th1(Rect(0, 0, 4, th1.rows - 1)) = uchar(0);
	//th1(Rect(th1.cols - 4, 0,4, th1.rows - 1)) = uchar(0);

	if (num == 0)
	{
		th1(Rect(0, 0, th1.cols - 1, 50)) = uchar(0);
	}
	else if (num == 1)
	{
		th1(Rect(0, th1.rows - 50, th1.cols - 1, 50)) = uchar(0);
	}

	findContours(th1, contours, CV_RETR_LIST, CHAIN_APPROX_SIMPLE);
	std::sort(contours.begin(), contours.end(), compareContourAreas);

	vector<Rect> boundRect(contours.size());
	vector<RotatedRect>box(contours.size());
	Point2f rect[4];
	for (vector<int>::size_type i = 0; i < contours.size(); i++)
	{
		Mat temp_mask = Mat::zeros(th1.rows, th1.cols, CV_8UC1);
		drawContours(temp_mask, contours, i, 255, FILLED, 8);
		double area = contourArea(contours[i]);
		if (area > 150 && area < 80000)
		{
			boundRect[i] = boundingRect(Mat(contours[i]));
			box[i] = minAreaRect(Mat(contours[i]));
			box[i].points(rect);
			float Width = sqrt(abs(rect[0].x - rect[1].x) * abs(rect[0].x - rect[1].x) + abs(rect[0].y - rect[1].y) * abs(rect[0].y - rect[1].y));
			float Height = sqrt(abs(rect[1].x - rect[2].x) * abs(rect[1].x - rect[2].x) + abs(rect[1].y - rect[2].y) * abs(rect[1].y - rect[2].y));
			float w = boundRect[i].width;
			float h = boundRect[i].height;
			RotatedRect rect = minAreaRect(contours[i]);  //包覆轮廓的最小斜矩形 划伤缺陷有旋转特点
			int X_1 = boundRect[i].tl().x;//矩形左上角X坐标值
			int Y_1 = boundRect[i].tl().y;//矩形左上角Y坐标值
			int X_2 = boundRect[i].br().x;//矩形右下角X坐标值
			int Y_2 = boundRect[i].br().y;//矩形右下角Y坐标值
			//int x_point = X_1 + round(w / 2);
			//int y_point = Y_1 + round(h / 2);
			double HeWid = max(Height / Width, Width / Height);
			if ((w < 5 && X_1 >= th1.cols - 4) || (w < 5 && X_2 <= 4))
			{
				continue;
			}
			//if (min(Height, Width) >= 200 && ((X_1 == 0 && Y_1 == 0) || (X_1 == 0 && Y_2 >= th1.rows - 1) || (Y_1 == 0 && X_2 >= th1.cols - 1) || (X_2 >= th1.cols - 1 && Y_2 >= th1.rows - 1)))
			//{
			//	continue;
			//}
			if (HeWid >= 3.2 && max(Height, Width) >= 40)
			{
				int border = 25;
				X_1 = X_1 - border;
				Y_1 = Y_1 - border;
				X_2 = X_2 + border;
				Y_2 = Y_2 + border;
				if (X_1 < 0)
				{
					X_1 = 0;
				}
				if (Y_1 < 0)
				{
					Y_1 = 0;
				}
				if (X_2 > th1.cols - 1)
				{
					X_2 = th1.cols - 1;
				}
				if (Y_2 > th1.rows - 1)
				{
					Y_2 = th1.rows - 1;
				}
				Mat ImageOutBinary;
				Mat tempImage = ImageBinary(Rect(X_1, Y_1, X_2 - X_1, Y_2 - Y_1));
				Mat tempBinary1 = temp_mask(Rect(X_1, Y_1, X_2 - X_1, Y_2 - Y_1)).clone();
				Mat tempBinary2 = th2(Rect(X_1, Y_1, X_2 - X_1, Y_2 - Y_1)).clone();
				Mat tempGray = img_gray(Rect(X_1, Y_1, X_2 - X_1, Y_2 - Y_1));

				double mean_all = mean(tempGray, tempImage)[0];
				threshold(tempGray, ImageOutBinary, mean_all - 10, 255, CV_THRESH_BINARY);
				double mean_In;
				mean_In = mean(tempGray, tempBinary1)[0];
				bitwise_and(ImageOutBinary, ~tempBinary2, tempBinary2);
				double mean_Out = mean(tempGray, tempBinary2)[0];
				double differ = mean_Out - mean_In;
				double MinmeanIn;
				if (mean_all >= 150)
					MinmeanIn = mean_all / 2 + 20;
				else if (mean_all >= 100)
					MinmeanIn = mean_all / 2 + 10;
				else
					MinmeanIn = mean_all / 2;
				if (((HeWid <= 8 && differ >= 13) || (HeWid > 8 && differ >= 9.5)) && mean_In >= MinmeanIn)
				{
					result = true;
					CvPoint top_lef4 = cvPoint(X_1, Y_1);
					CvPoint bottom_right4 = cvPoint(X_2, Y_2);
					rectangle(white, top_lef4, bottom_right4, Scalar(255, 255, 255), 5, 8, 0);
					break;
				}
			}
		}
	}
	if (result == true)
	{
		*mresult = white;
		*causecolor = "移位";
		result = true;
		imwrite("D:\\result_img_gray.bmp", img_gray);
		imwrite("D:\\result.bmp", white);
	}
	return result;
}
/*=========================================================
*@函 数 名: adaptiveThresholdCustom
*@功能描述: 自适应阈值分割实现图像二值化
*@param src          输入灰度图像
*@param dst          输出二值图像
*@param maxValue     输入满足阈值条件时像素取值
*@param method       计算局部均值方法
*@param type         输入阈值判断类型
*@param blockSize    卷积窗口大小(奇数)
*@param delta        输入偏移常量
*@param ratio        输入均值比例系数
*@备注说明：
=========================================================*/
//(img_gray, th1, 255, 0, 1, 51, 5.5, 1, 0.5)
/*=========================================================
 *@函 数 名: adaptiveThresholdCustom
 *@功能描述: 自适应阈值分割实现图像二值化
 *@param src          输入灰度图像
 *@param dst          输出二值图像
 *@param maxValue     输入满足阈值条件时像素取值
 *@param method       计算局部均值方法
 *@param type         输入阈值判断类型
 *@param blockSize    卷积窗口大小(奇数)
 *@param delta        输入偏移常量
 *@param ratio        输入均值比例系数
 *@备注说明：
 =========================================================*/
void adaptiveThresholdCustom(const cv::Mat &src, cv::Mat &dst, double maxValue, int method, int type, int blockSize, double delta, double ratio)
{
	CV_Assert(src.type() == CV_8UC1);               // 原图必须是单通道无符号8位,CV_Assert（）若括号中的表达式值为false，则返回一个错误信息
	CV_Assert(blockSize % 2 == 1 && blockSize > 1);	// 块大小必须大于1，并且是奇数
	CV_Assert(maxValue > 0);                        //二值图像最大值
	CV_Assert(ratio > DBL_EPSILON);	                //输入均值比例系数
	Size size = src.size();							//源图像的尺寸
	Mat _dst(size, src.type());						//目标图像的尺寸
	Mat mean;	                                    //存放均值图像
	if (src.data != _dst.data)
		mean = _dst;


	int top = (blockSize - 1)*0.5;     //填充的上边界行数
	int bottom = (blockSize - 1)*0.5;  //填充的下边界行数
	int left = (blockSize - 1)*0.5;	   //填充的左边界行数
	int right = (blockSize - 1)*0.5;   //填充的右边界行数
	int border_type = BORDER_CONSTANT; //边界填充方式
	Mat src_Expand;	                   //对原图像进行边界扩充

	Mat topImage = src(Rect(0, 0, src.cols, 1));//上边界一行图像

	cv::Scalar color = cv::mean(topImage)*0.5;//35-80之间均可以  该值需要确定

	//Scalar color = Scalar(50);//35-80之间均可以
	copyMakeBorder(src, src_Expand, top, bottom, left, right, border_type, color);

	if (method == ADAPTIVE_THRESH_MEAN_C)
	{
		/*
		@param src 单通道灰度图
		@param dst 单通道处理后的图
		@param int类型的ddepth，输出图像的深度
		@param Size类型的ksize，内核的大小
		@param Point类型的anchor，表示锚点
		@param bool类型的normaliz,即是否归一化
		@param borderType 图像外部像素的某种边界模式
		*/
		boxFilter(src_Expand, mean, src.type(), Size(blockSize, blockSize), Point(-1, -1), true, BORDER_CONSTANT);
	}
	else if (method == ADAPTIVE_THRESH_GAUSSIAN_C)
	{
		GaussianBlur(src, mean, Size(blockSize, blockSize), 0, 0, BORDER_DEFAULT);
	}
	else
		CV_Error(CV_StsBadFlag, "Unknown/unsupported adaptive threshold method");

	mean = mean(cv::Rect(top, top, src_Expand.cols - top * 2, src_Expand.rows - top * 2)); //删除扩充的图像边界

	int i, j;
	uchar imaxval = saturate_cast<uchar>(maxValue);	                       //将maxValue由double类型转换为uchar型
	int idelta = type == THRESH_BINARY ? cvCeil(delta) : cvFloor(delta);   //将idelta由double类型转换为int型
	if (src.isContinuous() && mean.isContinuous() && _dst.isContinuous())
	{
		size.width *= size.height;
		size.height = 1;
	}
	for (i = 0; i < size.height; i++)
	{
		const uchar* sdata = src.data + src.step * i;		   //指向源图像
		const uchar* mdata = mean.data + mean.step * i;		   //指向均值图
		uchar* ddata = _dst.data + _dst.step * i;	           //指向输出图
		for (j = 0; j < size.width; j++)
		{
			double Thresh = mdata[j] * ratio - idelta;	        //阈值
			if (CV_THRESH_BINARY == type)	                    //S>T时为imaxval
			{
				ddata[j] = sdata[j] > Thresh ? imaxval : 0;
			}
			else if (CV_THRESH_BINARY_INV == type)	            //S<T时为imaxval
			{
				ddata[j] = sdata[j] > Thresh ? 0 : imaxval;
			}
			else
				CV_Error(CV_StsBadFlag, "Unknown/unsupported threshold type");
		}
	}
	dst = _dst.clone();
}

/*=========================================================
* 函 数 名: Gabor7
* 功能描述: gabor滤波
=========================================================*/
Mat Gabor7(Mat img_1)
{
	Mat kernel1 = getGaborKernel(Size(5, 5), 1.1, CV_PI / 2, 1.0, 1.0, 0, CV_32F);//求卷积核
	float sum = 0.0;
	for (int i = 0; i < kernel1.rows; i++)
	{
		for (int j = 0; j < kernel1.cols; j++)
		{
			sum = sum + kernel1.ptr<float>(i)[j];
		}
	}
	Mat mmm = kernel1 / sum;
	Mat kernel2 = getGaborKernel(Size(5, 5), 1.1, 0, 1.0, 1.0, 0, CV_32F);
	float sum2 = 0.0;
	for (int i = 0; i < kernel2.rows; i++)
	{
		for (int j = 0; j < kernel2.cols; j++)
		{
			sum2 = sum2 + kernel2.ptr<float>(i)[j];
		}
	}
	Mat mmm2 = kernel2 / sum2;
	Mat img_4, img_5;
	filter2D(img_1, img_4, CV_8UC3, mmm);//卷积运算
	filter2D(img_4, img_5, CV_8UC3, mmm2);
	return img_5;
}