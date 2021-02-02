#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <cmath>
#include <iostream>
#include <time.h>
#include <string.h>
#include "PersTrans.h"
#include "time.h"//ͳ��ʱ������ӵ�ͷ�ļ�

using namespace cv;
using namespace std;

/*
�����������©����������ص��йز�����

©������ͣ�
1.��һ����С����޶� 3
2.��һ���������޶� 500
3.gaborͼ������Χ��С�ҶȲ��һ���ж��� 6.1
4.gaborͼ������Χ��С�ҶȲ�ڶ����ж��� 5.4
5.ԭͼͼ������Χ��С�ҶȲ�ڶ����ж��� 8.8
6.����Ӧ��ֵ��ֵ��������ƫ�Ƴ���     5.5

�������ͣ�
1.��һ����С����޶� 12  
2.��һ���������޶� 400
3.gaborͼ������Χ��С�ҶȲ��һ���ж��� 7.7
4.gaborͼ������Χ��С�ҶȲ�ڶ����ж��� 7.0
5.ԭͼͼ������Χ��С�ҶȲ�ڶ����ж��� 10.0
6.����Ӧ��ֵ��ֵ��������ƫ�Ƴ���     5.5

©������ͣ�����������
1.��һ����С����޶� 5
2.��һ���������޶� 400
3.gaborͼ������Χ��С�ҶȲ��һ���ж��� 5.9
4.gaborͼ������Χ��С�ҶȲ�ڶ����ж��� 5.2
5.ԭͼͼ������Χ��С�ҶȲ�ڶ����ж��� 8.6
6.����Ӧ��ֵ��ֵ��������ƫ�Ƴ���     5.5

˵��������6Խ�󣬶�ֵ���ָ�����İ�ɫ�������ԽС������Խ�٣�����ֵ��С������5.5-7.5֮��
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

	Mat src_L1 = imread("G:\\����Դ����\\20210111����\\��λ����\\23\\06_20200927165_23_112.bmp", -1);
	Mat src_R1 = imread("G:\\����Դ����\\20210111����\\��λ����\\23\\06_20200927165_23_012.bmp", -1);
	Mat src_ceguang_left = imread("G:\\����Դ����\\20210115����\\��λ���\\53\\10_20200927165_53_110.bmp", -1);
	Mat src_ceguang_right = imread("G:\\����Դ����\\20210115����\\��λ���\\53\\10_20200927165_53_010.bmp", -1);

	Mat M_L_1, M_R_1, M_L_1_E, M_R_1_E;
	if (src_L1.channels() == 3)
		cvtColor(src_L1, src_L1, CV_BGR2GRAY);
	if (src_R1.channels() == 3)
		cvtColor(src_R1, src_R1, CV_BGR2GRAY);
	if (src_ceguang_left.channels() == 3)
		cvtColor(src_ceguang_left, src_ceguang_left, CV_BGR2GRAY);
	if (src_ceguang_right.channels() == 3)
		cvtColor(src_ceguang_right, src_ceguang_right, CV_BGR2GRAY);
	//���ڰ��������
	bool Ext_Result_Left = f_LeftRightCam_PersTransMatCal(src_L1, &M_L_1, &M_L_1_E, "R��ˮ����", 15);
	bool Ext_Result_Right = f_LeftRightCam_PersTransMatCal(src_R1, &M_R_1, &M_R_1_E, "R��ˮ����", 15);

	Mat ceL1 = toushi_white(src_L1, M_L_1, -5, 3000, 1500);
	Mat ceR1 = toushi_white(src_R1, M_R_1, -5, 3000, 1500);
	Mat LeftCeGuang = toushi_white(src_ceguang_left, M_L_1, -5, 3000, 1500);      //��������У��ͼ
	Mat RightCeGuang = toushi_white(src_ceguang_right, M_R_1, -5, 3000, 1500);    //��������У��ͼ

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
	//gabor�˲�
	Mat leftfilter = Gabor7(ceL1);       //���׵��˲�
	Mat rightfilter = Gabor7(ceR1);     //�Ҳ�׵��˲�

	result = Shifting(leftfilter, &Mresult_1_white, &causeColor_1_white,1);
	if (!result)
	{
		result = Shifting(rightfilter, &Mresult_1_white, &causeColor_1_white,0);
	}

	cout << result << endl;
}

//�ȽϺ�������
bool compareContourAreas(std::vector< cv::Point> contour1,std::vector< cv::Point> contour2)
{
	return (cv::contourArea(contour1) > cv::contourArea(contour2));
}

/*====================================================================
* �� �� ��: Shifting
* ��������:��λ������Ϊ�׵�ͼ����һ������
* ���룺������׵�ͼ��
* �����������׵��¼����ͼ��result
* ������
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
			RotatedRect rect = minAreaRect(contours[i]);  //������������Сб���� ����ȱ������ת�ص�
			int X_1 = boundRect[i].tl().x;//�������Ͻ�X����ֵ
			int Y_1 = boundRect[i].tl().y;//�������Ͻ�Y����ֵ
			int X_2 = boundRect[i].br().x;//�������½�X����ֵ
			int Y_2 = boundRect[i].br().y;//�������½�Y����ֵ
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
		*causecolor = "��λ";
		result = true;
		imwrite("D:\\result_img_gray.bmp", img_gray);
		imwrite("D:\\result.bmp", white);
	}
	return result;
}
/*=========================================================
*@�� �� ��: adaptiveThresholdCustom
*@��������: ����Ӧ��ֵ�ָ�ʵ��ͼ���ֵ��
*@param src          ����Ҷ�ͼ��
*@param dst          �����ֵͼ��
*@param maxValue     ����������ֵ����ʱ����ȡֵ
*@param method       ����ֲ���ֵ����
*@param type         ������ֵ�ж�����
*@param blockSize    ������ڴ�С(����)
*@param delta        ����ƫ�Ƴ���
*@param ratio        �����ֵ����ϵ��
*@��ע˵����
=========================================================*/
//(img_gray, th1, 255, 0, 1, 51, 5.5, 1, 0.5)
/*=========================================================
 *@�� �� ��: adaptiveThresholdCustom
 *@��������: ����Ӧ��ֵ�ָ�ʵ��ͼ���ֵ��
 *@param src          ����Ҷ�ͼ��
 *@param dst          �����ֵͼ��
 *@param maxValue     ����������ֵ����ʱ����ȡֵ
 *@param method       ����ֲ���ֵ����
 *@param type         ������ֵ�ж�����
 *@param blockSize    ������ڴ�С(����)
 *@param delta        ����ƫ�Ƴ���
 *@param ratio        �����ֵ����ϵ��
 *@��ע˵����
 =========================================================*/
void adaptiveThresholdCustom(const cv::Mat &src, cv::Mat &dst, double maxValue, int method, int type, int blockSize, double delta, double ratio)
{
	CV_Assert(src.type() == CV_8UC1);               // ԭͼ�����ǵ�ͨ���޷���8λ,CV_Assert�����������еı��ʽֵΪfalse���򷵻�һ��������Ϣ
	CV_Assert(blockSize % 2 == 1 && blockSize > 1);	// ���С�������1������������
	CV_Assert(maxValue > 0);                        //��ֵͼ�����ֵ
	CV_Assert(ratio > DBL_EPSILON);	                //�����ֵ����ϵ��
	Size size = src.size();							//Դͼ��ĳߴ�
	Mat _dst(size, src.type());						//Ŀ��ͼ��ĳߴ�
	Mat mean;	                                    //��ž�ֵͼ��
	if (src.data != _dst.data)
		mean = _dst;


	int top = (blockSize - 1)*0.5;     //�����ϱ߽�����
	int bottom = (blockSize - 1)*0.5;  //�����±߽�����
	int left = (blockSize - 1)*0.5;	   //������߽�����
	int right = (blockSize - 1)*0.5;   //�����ұ߽�����
	int border_type = BORDER_CONSTANT; //�߽���䷽ʽ
	Mat src_Expand;	                   //��ԭͼ����б߽�����

	Mat topImage = src(Rect(0, 0, src.cols, 1));//�ϱ߽�һ��ͼ��

	cv::Scalar color = cv::mean(topImage)*0.5;//35-80֮�������  ��ֵ��Ҫȷ��

	//Scalar color = Scalar(50);//35-80֮�������
	copyMakeBorder(src, src_Expand, top, bottom, left, right, border_type, color);

	if (method == ADAPTIVE_THRESH_MEAN_C)
	{
		/*
		@param src ��ͨ���Ҷ�ͼ
		@param dst ��ͨ��������ͼ
		@param int���͵�ddepth�����ͼ������
		@param Size���͵�ksize���ں˵Ĵ�С
		@param Point���͵�anchor����ʾê��
		@param bool���͵�normaliz,���Ƿ��һ��
		@param borderType ͼ���ⲿ���ص�ĳ�ֱ߽�ģʽ
		*/
		boxFilter(src_Expand, mean, src.type(), Size(blockSize, blockSize), Point(-1, -1), true, BORDER_CONSTANT);
	}
	else if (method == ADAPTIVE_THRESH_GAUSSIAN_C)
	{
		GaussianBlur(src, mean, Size(blockSize, blockSize), 0, 0, BORDER_DEFAULT);
	}
	else
		CV_Error(CV_StsBadFlag, "Unknown/unsupported adaptive threshold method");

	mean = mean(cv::Rect(top, top, src_Expand.cols - top * 2, src_Expand.rows - top * 2)); //ɾ�������ͼ��߽�

	int i, j;
	uchar imaxval = saturate_cast<uchar>(maxValue);	                       //��maxValue��double����ת��Ϊuchar��
	int idelta = type == THRESH_BINARY ? cvCeil(delta) : cvFloor(delta);   //��idelta��double����ת��Ϊint��
	if (src.isContinuous() && mean.isContinuous() && _dst.isContinuous())
	{
		size.width *= size.height;
		size.height = 1;
	}
	for (i = 0; i < size.height; i++)
	{
		const uchar* sdata = src.data + src.step * i;		   //ָ��Դͼ��
		const uchar* mdata = mean.data + mean.step * i;		   //ָ���ֵͼ
		uchar* ddata = _dst.data + _dst.step * i;	           //ָ�����ͼ
		for (j = 0; j < size.width; j++)
		{
			double Thresh = mdata[j] * ratio - idelta;	        //��ֵ
			if (CV_THRESH_BINARY == type)	                    //S>TʱΪimaxval
			{
				ddata[j] = sdata[j] > Thresh ? imaxval : 0;
			}
			else if (CV_THRESH_BINARY_INV == type)	            //S<TʱΪimaxval
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
* �� �� ��: Gabor7
* ��������: gabor�˲�
=========================================================*/
Mat Gabor7(Mat img_1)
{
	Mat kernel1 = getGaborKernel(Size(5, 5), 1.1, CV_PI / 2, 1.0, 1.0, 0, CV_32F);//������
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
	filter2D(img_1, img_4, CV_8UC3, mmm);//�������
	filter2D(img_4, img_5, CV_8UC3, mmm2);
	return img_5;
}