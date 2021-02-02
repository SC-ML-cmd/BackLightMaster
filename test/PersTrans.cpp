#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <cmath>
#include <iostream>
#include <string.h>

using namespace cv;
using namespace std;

bool Ext_Result_Left_Right;
bool Ext_Result_Front_Back;
bool isArea_1, isArea_2;														//显示异常标志位
String Screen_Type = "R角水滴屏";

Mat toushi_white(Mat image, Mat M, int border, int length, int width);
bool f_MainCam_PersTransMatCal(InputArray _src, int border_white, int border_biankuang, Mat *Mwhite, Mat *Mbiankuang, Mat *M_white_abshow, int ID, String ScreenType_Flag);
Point2f getPointSlopeCrossPoint(Vec4f LineA, Vec4f LineB);
bool f_LeftRightCam_PersTransMatCal(InputArray _src, Mat *Mwhite, Mat *M_R_1_E, String ScreenType_Flag, int border_white);
bool f_FrontBackCam_PersTransMatCal(InputArray _src, Mat *Mwhite, String ScreenType_Flag);

/*=========================================================
* 函 数 名: toushi_white
* 功能描述: 透视变换图像矫正
=========================================================*/
Mat toushi_white(Mat image, Mat M, int border, int length, int width)
{
	Mat perspective;
	cv::warpPerspective(image, perspective, M, cv::Size(length, width), cv::INTER_LINEAR);
	return perspective;
}


/*=========================================================
*@函 数 名:              f_MainCam_PersTransMatCal
*@功能描述:              主黑白/彩色相机R角屏幕的透视变换矩阵计算
*@param _src             输入灰度/彩色图像
*@param _dst             输出显示到客户用图像
*@param border_white     提白底图边缘调整参数值
*@param border_black     提黑底图边缘调整参数值
*@param border_lightleak 提漏光图边缘调整参数值
*@param Mwhite           白底透视变换矩阵
*@param Mblack           黑底透视变换矩阵
*@param Mlightleak       漏光透视变换矩阵
*@param M_white_abshow   显示异常变换矩阵
*@param ID               工位ID号(弃用)
*@ScreenType_Flag        屏幕类型
*@修改时间：		     2020年9月18日
*@备注说明              use
=========================================================*/
bool f_MainCam_PersTransMatCal(InputArray _src, int border_white, int border_biankuang, Mat *Mwhite, Mat *Mbiankuang, Mat *M_white_abshow, int ID, String ScreenType_Flag)
{
	//    double screen_long=size_long/size_width;
	//    int screen_long=size_long/size_width;
	bool isArea_1, isArea_2;														//显示异常标志位
	Mat src = _src.getMat();                                                        //输入源图像
	if (src.type() == CV_8UC1)														//若输入8位图
		src = src.clone();															//拷贝原图
	else
		cvtColor(src, src, CV_BGR2GRAY);										    //灰度化彩色图
	CV_Assert(src.depth() == CV_8U);                                                //8位无符号
	Mat binaryImage = Mat::zeros(src.size(), CV_8UC1);                              //二值图像
	threshold(src, binaryImage, 40, 255, CV_THRESH_BINARY);							//二值化(有问题)
	medianBlur(binaryImage, binaryImage, 5);										//中值滤波去除锯齿
	int displayError_Areasignal = 0;												//根据轮廓面积判定显示异常标志位
	vector<vector<Point>> contours;													//contours存放点集信息
	findContours(binaryImage, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);    //CV_RETR_EXTERNAL最外轮廓，CV_CHAIN_APPROX_NONE所有轮廓点信息
	vector<Point> upLinePoint, leftLinePoint, downLinePoint, rightLinePoint;		//上左下右侧点集数据
	Vec4f upLine_Fit, leftLine_Fit, downLine_Fit, rightLine_Fit;				    //上左下右拟合直线数据
	vector<Point2f> src_corner(4), src_corner_biankuang(4), src_corner_abshow(4);   //四个边相交得到角点坐标，漏光角点，显示异常角点
	Rect rect;																        //最小正外接矩形
	int x1, y1, x2, y2, x3, y3, x4, y4;			//正接矩阵坐标点信息
	for (vector<int>::size_type i = 0; i < contours.size(); i++)
	{
		double area = contourArea(contours[i]);
		if (area > 2500000 && area < 5000000)
		{
			displayError_Areasignal++;
			rect = boundingRect(contours[i]);
			x1 = rect.tl().x;//左上角
			y1 = rect.tl().y;//左上角
			x2 = rect.tl().x;//左下角
			y2 = rect.br().y;//右下角
			x3 = rect.br().x;//右下角
			y3 = rect.br().y;//右下角
			x4 = rect.br().x;//右上角
			y4 = rect.tl().y;//右上角
			int radianEliminate = 230;
			int deviation = 160;
			for (int j = 0; j < contours[i].size(); j++)
			{
				//左侧点集
				if (contours[i][j].y > y1 + radianEliminate && contours[i][j].y < y1 + (y2 - y1)*0.3 && abs(contours[i][j].x - x1) < deviation ||
					contours[i][j].y > y1 + (y2 - y1)*0.7 && contours[i][j].y < y2 - radianEliminate && abs(contours[i][j].x - x1) < deviation)
					leftLinePoint.push_back(Point(contours[i][j].x, contours[i][j].y));
				//右侧点集
				if (contours[i][j].y > y1 + radianEliminate && contours[i][j].y < y2 - radianEliminate && abs(contours[i][j].x - x3) < deviation)
					rightLinePoint.push_back(Point(contours[i][j].x, contours[i][j].y));
				//上侧点集
				if (contours[i][j].x > x1 + radianEliminate && contours[i][j].x < x4 - radianEliminate && abs(contours[i][j].y - y1) < deviation)
					upLinePoint.push_back(Point(contours[i][j].x, contours[i][j].y));
				//下侧点集
				if (contours[i][j].x > x1 + radianEliminate && contours[i][j].x < x4 - radianEliminate && abs(contours[i][j].y - y2) < deviation)
					downLinePoint.push_back(Point(contours[i][j].x, contours[i][j].y));
			}
		}
	}
	if (leftLinePoint.size() == 0 || rightLinePoint.size() == 0 || upLinePoint.size() == 0 || downLinePoint.size() == 0)
		displayError_Areasignal = 0;
	//根据轮廓面积判定显示异常
	if (displayError_Areasignal > 0 && ID == 1)
		isArea_1 = false;
	if (displayError_Areasignal == 0 && ID == 1)
		isArea_1 = true;
	if (displayError_Areasignal > 0 && ID == 2)
		isArea_2 = false;
	if (displayError_Areasignal == 0 && ID == 2)
		isArea_2 = true;
	//未提取到屏幕判定显示异常提取边缘角落
	if (displayError_Areasignal == 0)
	{
		vector<Point2f> src_points(4);
		src_points = { Point(0, 0), Point(0, 10), Point(10, 10), Point(10, 0) };
		vector<Point2f> dst_points(4);
		if (ScreenType_Flag == "矩形屏")
			dst_points = { Point(0, 0), Point(0, 1775), Point(3000, 1775), Point(3000, 0) };
		else        //pixel_num
			//dst_points = { Point(0, 0), Point(0, 1500), Point(3000, 1500), Point(3000, 0) };
			dst_points = { Point(0, 0), Point(0, 1500), Point(3000, 1500), Point(3000, 0) };
		*Mwhite = cv::getPerspectiveTransform(src_points, dst_points);
		//*Mlightleak = cv::getPerspectiveTransform(src_points, dst_points);
		*Mbiankuang = cv::getPerspectiveTransform(src_points, dst_points);
		*M_white_abshow = cv::getPerspectiveTransform(src_points, dst_points);
	}
	//正常屏幕提取屏幕的四个角点
	else
	{
		fitLine(leftLinePoint, leftLine_Fit, cv::DIST_L2, 0, 1e-2, 1e-2);               //左侧拟合直线
		fitLine(rightLinePoint, rightLine_Fit, cv::DIST_L2, 0, 1e-2, 1e-2);				//右侧拟合直线
		fitLine(upLinePoint, upLine_Fit, cv::DIST_L2, 0, 1e-2, 1e-2);					//上侧拟合直线
		fitLine(downLinePoint, downLine_Fit, cv::DIST_L2, 0, 1e-2, 1e-2);				//下侧拟合直线

		/*Mat img = _src.getMat();
		for (int i = 0; i < leftLinePoint.size(); i++)
		{
			circle(img, Point(leftLinePoint[i].x, leftLinePoint[i].y), 8, Scalar(0, 0, 255), -1);
		}
		for (int i = 0; i < rightLinePoint.size(); i++)
		{
			circle(img, Point(rightLinePoint[i].x, rightLinePoint[i].y), 8, Scalar(0, 0, 255), -1);
		}
		for (int i = 0; i < upLinePoint.size(); i++)
		{
			circle(img, Point(upLinePoint[i].x, upLinePoint[i].y), 8, Scalar(0, 0, 255), -1);
		}
		for (int i = 0; i < downLinePoint.size(); i++)
		{
			circle(img, Point(downLinePoint[i].x, downLinePoint[i].y), 8, Scalar(0, 0, 255), -1);
		}*/

		src_corner[0] = getPointSlopeCrossPoint(upLine_Fit, leftLine_Fit);              //左上角点
		src_corner[1] = getPointSlopeCrossPoint(downLine_Fit, leftLine_Fit);		    //左下角点
		src_corner[2] = getPointSlopeCrossPoint(downLine_Fit, rightLine_Fit);	        //右下角点
		src_corner[3] = getPointSlopeCrossPoint(upLine_Fit, rightLine_Fit);		        //右上角点

																						//对4个角点的坐标位置进行微调（白底图以及黑底图）
		src_corner[0].x = src_corner[0].x - border_white;
		src_corner[0].y = src_corner[0].y - border_white;
		src_corner[1].x = src_corner[1].x - border_white;
		src_corner[1].y = src_corner[1].y + border_white;
		src_corner[2].x = src_corner[2].x + border_white;
		src_corner[2].y = src_corner[2].y + border_white;
		src_corner[3].x = src_corner[3].x + border_white;
		src_corner[3].y = src_corner[3].y - border_white;
		//对4个角点的坐标位置进行微调（漏光检测图）
		src_corner_biankuang[0].x = src_corner[0].x - border_biankuang;
		src_corner_biankuang[0].y = src_corner[0].y - border_biankuang;
		src_corner_biankuang[1].x = src_corner[1].x - border_biankuang;
		src_corner_biankuang[1].y = src_corner[1].y + border_biankuang;
		src_corner_biankuang[2].x = src_corner[2].x + border_biankuang;
		src_corner_biankuang[2].y = src_corner[2].y + border_biankuang;
		src_corner_biankuang[3].x = src_corner[3].x + border_biankuang;
		src_corner_biankuang[3].y = src_corner[3].y - border_biankuang;
		//显示异常(白底图)
		src_corner_abshow[0].x = src_corner[0].x - border_white + 10;
		src_corner_abshow[0].y = src_corner[0].y - border_white + 10;
		src_corner_abshow[1].x = src_corner[1].x - border_white + 10;
		src_corner_abshow[1].y = src_corner[1].y + border_white - 10;
		src_corner_abshow[2].x = src_corner[2].x + border_white - 10;
		src_corner_abshow[2].y = src_corner[2].y + border_white - 10;
		src_corner_abshow[3].x = src_corner[3].x + border_white - 10;
		src_corner_abshow[3].y = src_corner[3].y - border_white + 10;

		vector<Point2f> dst_corner(4);
		if (ScreenType_Flag == "矩形屏")
		{
			dst_corner[0] = Point(0, 0);
			dst_corner[1] = Point(0, 1775);
			dst_corner[2] = Point(3000, 1775);
			dst_corner[3] = Point(3000, 0);
		}
		else
		{
			dst_corner[0] = Point(0, 0);
			dst_corner[1] = Point(0, 1500);
			//            dst_corner[2] = Point(3000, 1500);
			//            dst_corner[3] = Point(3000, 0);
			dst_corner[2] = Point(3000, 1500);
			dst_corner[3] = Point(3000, 0);
		}
		*Mwhite = cv::getPerspectiveTransform(src_corner, dst_corner);
		//*Mlightleak = cv::getPerspectiveTransform(src_corner_lightleak, dst_corner);
		*Mbiankuang = cv::getPerspectiveTransform(src_corner_biankuang, dst_corner);
		*M_white_abshow = cv::getPerspectiveTransform(src_corner_abshow, dst_corner);
	}
	if (ID == 1)
		return isArea_1;
	else
		return isArea_2;
}

/*=========================================================
*@函 数 名:     getPointSlopeCrossPoint
*@功能描述:     计算点斜式两条直线的交点
*@param LineA   平行线条
*@param LineB   垂直线条
*@编制时间：    2020年8月17日
*@备注说明
=========================================================*/
Point2f getPointSlopeCrossPoint(Vec4f LineA, Vec4f LineB)
{
	const double PI = 3.1415926535897;
	Point2f crossPoint;
	double kA = LineA[1] / LineA[0];
	double kB = LineB[1] / LineB[0];
	double theta = atan2(LineB[1], LineB[0]);
	if (theta == PI * 0.5)
	{
		crossPoint.x = LineB[0];
		crossPoint.y = kA * LineB[0] + LineA[3] - kA * LineA[2];
		return crossPoint;
	}
	double bA = LineA[3] - kA * LineA[2];
	double bB = LineB[3] - kB * LineB[2];
	crossPoint.x = (bB - bA) / (kA - kB);
	crossPoint.y = (kA*bB - kB * bA) / (kA - kB);
	return crossPoint;
}

/*=========================================================
*@函 数 名:              f_LeftRightCam_PersTransMatCal
*@功能描述:              左右相机R角透视变换矩阵计算函数
*@param _src             输入灰度/彩色图像
*@param Mwhite           白底透视变换矩阵
*@ScreenType_Flag        屏幕类型
*@编制时间：		     2020年8月20日
*@备注说明              use
=========================================================*/
bool f_LeftRightCam_PersTransMatCal(InputArray _src, Mat *Mwhite, Mat *M_R_1_E, String ScreenType_Flag, int border_white)
{
	bool Ext_Result_Left_Right;                                                     //提取屏幕成功标志位
	Mat src = _src.getMat();                                                        //输入源图像
	if (src.type() == CV_8UC1)														//若输入8位图
		src = src.clone();															//拷贝原图
	else
		cvtColor(src, src, CV_BGR2GRAY);										    //灰度化彩色图
	CV_Assert(src.depth() == CV_8U);                                                //8位无符号
	Mat binaryImage = Mat::zeros(src.size(), CV_8UC1);                              //二值图像
	threshold(src, binaryImage, 40, 255, CV_THRESH_BINARY);							//二值化(有问题)
	medianBlur(binaryImage, binaryImage, 5);										//中值滤波去除锯齿
	int displayError_Areasignal = 0;												//根据轮廓面积判定显示异常标志位
	vector<vector<Point>> contours;													//contours存放点集信息
	findContours(binaryImage, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);    //CV_RETR_EXTERNAL最外轮廓，CV_CHAIN_APPROX_NONE所有轮廓点信息
	vector<Point> upLinePoint, leftLinePoint, downLinePoint, rightLinePoint;		//上左下右侧点集数据
	Vec4f upLine_Fit, leftLine_Fit, downLine_Fit, rightLine_Fit;				    //上左下右拟合直线数据
	vector<Point2f> src_corner(4);                                                  //四个边相交得到角点坐标
	vector<Point2f> src_corner_enlarge(4);
	Rect rect;																        //最小正外接矩形
	int x1, y1, x2, y2, x3, y3, x4, y4;			                                    //正接矩阵坐标点信息
	vector<Point2f> dst_corner(4);                                                  //透视变换后的点的信息
	for (vector<int>::size_type i = 0; i < contours.size(); i++)
	{
		double area = contourArea(contours[i]);
		if (area > 1500000 && area < 5000000)
		{
			displayError_Areasignal++;
			rect = cv::boundingRect(contours[i]);                                           //最小外接矩形提取
			Ext_Result_Left_Right = false;                                                             //提取到屏幕

			//cv::rectangle(src, rect, Scalar(255, 0, 0), 5, LINE_8, 0);
			x1 = rect.tl().x;//左上角
			y1 = rect.tl().y;//左上角
			x2 = rect.tl().x;//左下角
			y2 = rect.br().y;//左下角
			x3 = rect.br().x;//右下角
			y3 = rect.br().y;//右下角
			x4 = rect.br().x;//右上角
			y4 = rect.tl().y;//右上角
																									   //矩形面积缩小1/3，并得到新的矩形顶点
			//取直线的参数设置
			int radianEliminate = 230;//(R角)左右使用
			int radianEliminate2 = 360;//(R角)上下使用
			int deviation = 200;//(斜线带来的误差)左右使用
			int deviation2 = 120;//(斜线带来的误差)上下使用

			if (displayError_Areasignal != 0)
			{
				for (int j = 0; j < contours[i].size(); j++)
				{
					//左侧点集
					if (contours[i][j].y > y1 + radianEliminate && contours[i][j].y < y1 + (y2 - y1)*0.3 && abs(contours[i][j].x - x1) < deviation ||
						contours[i][j].y > y1 + (y2 - y1)*0.7 && contours[i][j].y < y2 - radianEliminate && abs(contours[i][j].x - x1) < deviation)
						leftLinePoint.push_back(Point(contours[i][j].x, contours[i][j].y));
					//右侧点集
					if (contours[i][j].y > y1 + radianEliminate && contours[i][j].y < y2 - radianEliminate && abs(contours[i][j].x - x3) < deviation)
						rightLinePoint.push_back(Point(contours[i][j].x, contours[i][j].y));
					//上侧点集
					if (contours[i][j].x > x1 + radianEliminate2 && contours[i][j].x < x4 - radianEliminate2 && abs(contours[i][j].y - y1) < deviation2)
						upLinePoint.push_back(Point(contours[i][j].x, contours[i][j].y));
					if (contours[i][j].x > x1 + radianEliminate2 && contours[i][j].x < x4 - radianEliminate2 && abs(contours[i][j].y - y2) < deviation2)
						downLinePoint.push_back(Point(contours[i][j].x, contours[i][j].y));
				}
				//下侧点集
				//downLinePoint.push_back(Point((x2 + x3) / 2, (y2 + y3) / 2));
				if (leftLinePoint.size() != 0 && rightLinePoint.size() != 0 && upLinePoint.size() != 0 && downLinePoint.size() != 0)
				{
					//直线拟合
					fitLine(leftLinePoint, leftLine_Fit, cv::DIST_L2, 0, 1e-2, 1e-2);               //左侧拟合直线
					fitLine(rightLinePoint, rightLine_Fit, cv::DIST_L2, 0, 1e-2, 1e-2);				//右侧拟合直线
					fitLine(upLinePoint, upLine_Fit, cv::DIST_L2, 0, 1e-2, 1e-2);					//上侧拟合直线
					downLine_Fit[0] = upLine_Fit[0];
					downLine_Fit[1] = upLine_Fit[1];
					downLine_Fit[2] = downLinePoint[0].x;
					downLine_Fit[3] = downLinePoint[0].y;                                           //下侧直线确定

					/*Mat img = _src.getMat();
					for (int i = 0; i < leftLinePoint.size(); i++)
					{
						circle(img, Point(leftLinePoint[i].x, leftLinePoint[i].y), 8, Scalar(0, 0, 255), -1);
					}
					for (int i = 0; i < rightLinePoint.size(); i++)
					{
						circle(img, Point(rightLinePoint[i].x, rightLinePoint[i].y), 8, Scalar(0, 0, 255), -1);
					}
					for (int i = 0; i < upLinePoint.size(); i++)
					{
						circle(img, Point(upLinePoint[i].x, upLinePoint[i].y), 8, Scalar(0, 0, 255), -1);
					}
					for (int i = 0; i < downLinePoint.size(); i++)
					{
						circle(img, Point(downLinePoint[i].x, downLinePoint[i].y), 8, Scalar(0, 0, 255), -1);
					}*/
					//角点提取
					src_corner[0] = getPointSlopeCrossPoint(upLine_Fit, leftLine_Fit);              //左上角点
					src_corner[1] = getPointSlopeCrossPoint(downLine_Fit, leftLine_Fit);		    //左下角点
					src_corner[2] = getPointSlopeCrossPoint(downLine_Fit, rightLine_Fit);	        //右下角点
					src_corner[3] = getPointSlopeCrossPoint(upLine_Fit, rightLine_Fit);		        //右上角点


					//src_corner_enlarge[0] = Point2f(xcoordinate1 + tl.x - border_white, ycoordinate1 + tl.y - border_white);	                         //左上角
					//src_corner_enlarge[1] = Point2f(xcoordinate2 + bl.x - border_white, ycoordinate2 - height / 3 + bl.y + border_white);              //左下角
					//src_corner_enlarge[2] = Point2f(xcoordinate3 - width / 4 + br.x + border_white, ycoordinate3 - height / 3 + br.y + border_white);	 //右下角
					//src_corner_enlarge[3] = Point2f(xcoordinate4 - width / 4 + tr.x + border_white, ycoordinate4 + tr.y - border_white);	             //右上角

					src_corner_enlarge[0].y = src_corner[0].y - border_white;
					src_corner_enlarge[0].x = src_corner[0].x - border_white;
					src_corner_enlarge[1].y = src_corner[1].y + border_white;
					src_corner_enlarge[1].x = src_corner[1].x - border_white;
					src_corner_enlarge[2].y = src_corner[2].y + border_white;
					src_corner_enlarge[2].x = src_corner[2].x + border_white;
					src_corner_enlarge[3].y = src_corner[3].y - border_white;
					src_corner_enlarge[3].x = src_corner[3].x + border_white;
					//透视变换矩阵计算
					if (ScreenType_Flag == "矩形屏")
						dst_corner = { Point(0, 0), Point(0, 1183), Point(3000, 1183), Point(3000, 0) };
					else
						dst_corner = { Point(0, 0), Point(0, 1500), Point(3000, 1500), Point(3000, 0) };
					*Mwhite = cv::getPerspectiveTransform(src_corner, dst_corner);
					*M_R_1_E = cv::getPerspectiveTransform(src_corner_enlarge, dst_corner);
				}
				else
				{
					displayError_Areasignal = 0;
					break;
				}
			}
		}
	}
	//没有提取到屏幕
	if (displayError_Areasignal == 0)
	{
		Ext_Result_Left_Right = true; //没有提取到屏幕
		vector<Point2f> src_points(4);
		src_points = { Point(0, 0), Point(0, 10), Point(10, 10), Point(10, 0) };
		vector<Point2f> dst_points(4);
		if (ScreenType_Flag == "矩形屏")
			dst_points = { Point(0, 0), Point(0, 1775), Point(3000, 1775), Point(3000, 0) };
		else
			dst_points = { Point(0, 0), Point(0, 1500), Point(3000, 1500), Point(3000, 0) };
		*Mwhite = cv::getPerspectiveTransform(src_points, dst_points);                        //透视变换矩阵提取
		*M_R_1_E = cv::getPerspectiveTransform(src_points, dst_points);
	}

	return Ext_Result_Left_Right;
}

/*=========================================================
*@函 数 名:              f_FrontBackCam_PersTransMatCal
*@功能描述:              前后相机R角透视变换矩阵计算函数
*@param _src             输入灰度/彩色图像
*@param Mwhite           白底透视变换矩阵
*@ScreenType_Flag        屏幕类型
*@编制时间：		     2020年8月21日
*@备注说明
=========================================================*/
bool f_FrontBackCam_PersTransMatCal(InputArray _src, Mat *Mwhite, String ScreenType_Flag)
{
	bool Ext_Result_Front_Back;                                                     //提取屏幕成功标志位
	Mat src = _src.getMat();                                                        //输入源图像
	if (src.type() == CV_8UC1)														//若输入8位图
		src = src.clone();															//拷贝原图
	else
		cvtColor(src, src, CV_BGR2GRAY);										    //灰度化彩色图
	CV_Assert(src.depth() == CV_8U);                                                //8位无符号
	Mat binaryImage = Mat::zeros(src.size(), CV_8UC1);                              //二值图像
	threshold(src, binaryImage, 40, 255, CV_THRESH_BINARY);							//二值化(有问题)
	medianBlur(binaryImage, binaryImage, 5);										//中值滤波去除锯齿
	int displayError_Areasignal = 0;												//根据轮廓面积判定显示异常标志位
	vector<vector<Point>> contours;													//contours存放点集信息
	findContours(binaryImage, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);    //CV_RETR_EXTERNAL最外轮廓，CV_CHAIN_APPROX_NONE所有轮廓点信息
	vector<Point> upLinePoint, leftLinePoint, downLinePoint, rightLinePoint;		//上左下右侧点集数据
	Vec4f upLine_Fit, leftLine_Fit, downLine_Fit, rightLine_Fit;				    //上左下右拟合直线数据
	vector<Point2f> src_corner(4);                                                  //四个边相交得到角点坐标
	Rect rect;																        //最小正外接矩形
	int x1, y1, x2, y2, x3, y3, x4, y4;			                                    //正接矩阵坐标点信息
	vector<Point2f> dst_corner(4);                                                  //透视变换后的点的信息
	for (vector<int>::size_type i = 0; i < contours.size(); i++)
	{
		double area = contourArea(contours[i]);
		if (area > 2000000 && area < 5000000)
		{
			displayError_Areasignal++;
			rect = cv::boundingRect(contours[i]);                                           //最小外接矩形提取
			Ext_Result_Front_Back = false;                                                             //提取到屏幕
			//矩形面积缩小1/3，并得到新的矩形顶点
			int PixelGap1 = rect.tl().x;
			int PixelGap2 = src.cols - (rect.tl().x + rect.width);
			//横坐标获取
			if (PixelGap1 > PixelGap2)
			{
				x1 = rect.tl().x;
				x2 = x1;
				x3 = (rect.br().x - x1) * 3 / 5 + x1;
				x4 = x3;
			}
			else
			{
				x3 = rect.br().x;
				x4 = x3;
				x1 = x3 - (x3 - rect.tl().x) * 3 / 5;
				x2 = x1;
			}
			//纵坐标获取
			y1 = rect.tl().y;
			y2 = rect.br().y;
			if (y2 >= src.rows)
				y2 = src.rows - 1;
			y3 = y2;
			y4 = y1;
			//取直线的参数设置
			int radianEliminate = 350;//(R角)左右使用
			int radianEliminate1 = 480;//(R角)左右使用
			int radianEliminate2 = 230;//(R角)上下使用
			int deviation = 120;//(斜线带来的误差)左右使用
			int deviation2 = 200;//(斜线带来的误差)上下使用
			if (PixelGap1 > PixelGap2)
			{
				//外接矩形缩小
				while (binaryImage.at<uchar>(y3, x3) == 255 || binaryImage.at<uchar>(y4, x4) == 255)
				{
					x3 = x3 - 1;
					x4 = x4 - 1;

					if (x3 == 0 || x4 == 0)
					{
						displayError_Areasignal = 0;
						break;
					}
				}
				while (binaryImage.at<uchar>(y3, x3) != 255)
				{
					y3 = y3 - 1;
					y2 = y3;

					if (y2 == 0 || y3 == 0)
					{
						displayError_Areasignal = 0;
						break;
					}
				}
				while (binaryImage.at<uchar>(y4, x4) != 255)
				{
					y4 = y4 + 1;
					y1 = y4;

					if (y4 == src.rows - 1 || y1 == src.rows - 1)
					{
						displayError_Areasignal = 0;
						break;
					}
				}
				if (displayError_Areasignal != 0)
				{
					for (int j = 0; j < contours[i].size(); j++)
					{
						//左侧点集
						if (contours[i][j].y > y1 + radianEliminate && contours[i][j].y < y2 - radianEliminate1 && abs(contours[i][j].x - x1) < deviation)
							leftLinePoint.push_back(Point(contours[i][j].x, contours[i][j].y));
						//上侧点集
						if (contours[i][j].x > x1 + radianEliminate2 && contours[i][j].x < x4 - radianEliminate2 && abs(contours[i][j].y - y1) < deviation2)
							upLinePoint.push_back(Point(contours[i][j].x, contours[i][j].y));
						//下侧点集
						if (contours[i][j].x > x1 + radianEliminate2 && contours[i][j].x < x4 - radianEliminate2 && abs(contours[i][j].y - y2) < deviation2)
							downLinePoint.push_back(Point(contours[i][j].x, contours[i][j].y));
					}
					//右侧点集
					rightLinePoint.push_back(Point((x3 + x4) / 2, (y3 + y4) / 2));
					if (leftLinePoint.size() != 0 && upLinePoint.size() != 0 && downLinePoint.size() != 0)
					{
						//直线拟合
						fitLine(leftLinePoint, leftLine_Fit, cv::DIST_L2, 0, 1e-2, 1e-2);               //左侧拟合直线
						rightLine_Fit[0] = leftLine_Fit[0];
						rightLine_Fit[1] = leftLine_Fit[1];
						rightLine_Fit[2] = rightLinePoint[0].x;
						rightLine_Fit[3] = rightLinePoint[0].y;                                         //右侧拟合直线
						fitLine(upLinePoint, upLine_Fit, cv::DIST_L2, 0, 1e-2, 1e-2);					//上侧拟合直线
						fitLine(downLinePoint, downLine_Fit, cv::DIST_L2, 0, 1e-2, 1e-2);               //下侧拟合直线
						//角点提取
						src_corner[0] = getPointSlopeCrossPoint(upLine_Fit, leftLine_Fit);              //左上角点
						src_corner[1] = getPointSlopeCrossPoint(downLine_Fit, leftLine_Fit);		    //左下角点
						src_corner[2] = getPointSlopeCrossPoint(downLine_Fit, rightLine_Fit);	        //右下角点
						src_corner[3] = getPointSlopeCrossPoint(upLine_Fit, rightLine_Fit);		        //右上角点
						//透视变换矩阵计算
						if (ScreenType_Flag == "矩形屏")
							dst_corner = { Point(0, 0), Point(0, 1775), Point(2000, 1775), Point(2000, 0) };
						else
							dst_corner = { Point(0, 0), Point(0, 1500), Point(2000, 1500), Point(2000, 0) };
						*Mwhite = cv::getPerspectiveTransform(src_corner, dst_corner);
					}
					else
					{
						displayError_Areasignal = 0;
						break;
					}
				}
			}
			else
			{
				//外接矩形缩小
				while (binaryImage.at<uchar>(y1, x1) == 255 || binaryImage.at<uchar>(y2, x2) == 255)
				{
					x1 = x1 + 1;
					x2 = x2 + 1;

					if (x1 == src.cols - 1 || x1 == src.cols - 1)
					{
						displayError_Areasignal = 0;
						break;
					}
				}
				while (binaryImage.at<uchar>(y1, x1) != 255)
				{
					y1 = y1 + 1;
					y4 = y1;

					if (y1 == src.rows - 1 || y4 == src.rows - 1)
					{
						displayError_Areasignal = 0;
						break;
					}
				}
				while (binaryImage.at<uchar>(y2, x2) != 255)
				{
					y2 = y2 - 1;
					y3 = y2;

					if (y2 == 0 || y3 == 0)
					{
						displayError_Areasignal = 0;
						break;
					}
				}
				if (displayError_Areasignal != 0)
				{
					for (int j = 0; j < contours[i].size(); j++)
					{
						//右侧点集
						if (contours[i][j].y > y1 + radianEliminate1 && contours[i][j].y < y1 + (y2 - y1) * 0.3 &&abs(contours[i][j].x - x3) < deviation || contours[i][j].y > y1 + (y2 - y1)*0.7 && contours[i][j].y < y2 - radianEliminate && abs(contours[i][j].x - x3) < deviation)
							rightLinePoint.push_back(Point(contours[i][j].x, contours[i][j].y));
						//上侧点集
						if (contours[i][j].x > x1 + radianEliminate2 && contours[i][j].x < x4 - radianEliminate2 && abs(contours[i][j].y - y1) < deviation2)
							upLinePoint.push_back(Point(contours[i][j].x, contours[i][j].y));
						//下侧点集
						if (contours[i][j].x > x1 + radianEliminate2 && contours[i][j].x < x4 - radianEliminate2 && abs(contours[i][j].y - y2) < deviation2)
							downLinePoint.push_back(Point(contours[i][j].x, contours[i][j].y));
					}
					//左侧点集
					leftLinePoint.push_back(Point((x1 + x2) / 2, (y1 + y2) / 2));
					if (rightLinePoint.size() != 0 && upLinePoint.size() != 0 && downLinePoint.size() != 0)
					{
						//拟合直线
						fitLine(rightLinePoint, rightLine_Fit, cv::DIST_L2, 0, 1e-2, 1e-2);				//右侧拟合直线
						leftLine_Fit[0] = rightLine_Fit[0];
						leftLine_Fit[1] = rightLine_Fit[1];
						leftLine_Fit[2] = leftLinePoint[0].x;
						leftLine_Fit[3] = leftLinePoint[0].y;                                           //左侧拟合直线
						fitLine(upLinePoint, upLine_Fit, cv::DIST_L2, 0, 1e-2, 1e-2);				    //上侧拟合直线
						fitLine(downLinePoint, downLine_Fit, cv::DIST_L2, 0, 1e-2, 1e-2);				//下侧拟合直线
						//角点提取
						src_corner[0] = getPointSlopeCrossPoint(upLine_Fit, leftLine_Fit);              //左上角点
						src_corner[1] = getPointSlopeCrossPoint(downLine_Fit, leftLine_Fit);		    //左下角点
						src_corner[2] = getPointSlopeCrossPoint(downLine_Fit, rightLine_Fit);	        //右下角点
						src_corner[3] = getPointSlopeCrossPoint(upLine_Fit, rightLine_Fit);		        //右上角点
						//透视变换矩阵计算
						if (ScreenType_Flag == "矩形屏")
							dst_corner = { Point(0, 0), Point(0, 1775), Point(2000, 1775), Point(2000, 0) };
						else
							dst_corner = { Point(0, 0), Point(0, 1500), Point(2000, 1500), Point(2000, 0) };
						*Mwhite = cv::getPerspectiveTransform(src_corner, dst_corner);
					}
					else
					{
						displayError_Areasignal = 0;
						break;
					}
				}
			}
		}
	}
	//没有提取到屏幕
	if (displayError_Areasignal == 0)
	{
		Ext_Result_Front_Back = true; //没有提取到屏幕
		vector<Point2f> src_points(4);
		src_points = { Point(0, 0), Point(0, 10), Point(10, 10), Point(10, 0) };
		vector<Point2f> dst_points(4);
		if (ScreenType_Flag == "矩形屏")
			dst_points = { Point(0, 0), Point(0, 1775), Point(3000, 1775), Point(3000, 0) };
		else
			dst_points = { Point(0, 0), Point(0, 1500), Point(3000, 1500), Point(3000, 0) };
		*Mwhite = cv::getPerspectiveTransform(src_points, dst_points);                        //透视变换矩阵提取
	}

	return Ext_Result_Front_Back;
}




