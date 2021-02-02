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
bool isArea_1, isArea_2;														//��ʾ�쳣��־λ
String Screen_Type = "R��ˮ����";

Mat toushi_white(Mat image, Mat M, int border, int length, int width);
bool f_MainCam_PersTransMatCal(InputArray _src, int border_white, int border_biankuang, Mat *Mwhite, Mat *Mbiankuang, Mat *M_white_abshow, int ID, String ScreenType_Flag);
Point2f getPointSlopeCrossPoint(Vec4f LineA, Vec4f LineB);
bool f_LeftRightCam_PersTransMatCal(InputArray _src, Mat *Mwhite, Mat *M_R_1_E, String ScreenType_Flag, int border_white);
bool f_FrontBackCam_PersTransMatCal(InputArray _src, Mat *Mwhite, String ScreenType_Flag);

/*=========================================================
* �� �� ��: toushi_white
* ��������: ͸�ӱ任ͼ�����
=========================================================*/
Mat toushi_white(Mat image, Mat M, int border, int length, int width)
{
	Mat perspective;
	cv::warpPerspective(image, perspective, M, cv::Size(length, width), cv::INTER_LINEAR);
	return perspective;
}


/*=========================================================
*@�� �� ��:              f_MainCam_PersTransMatCal
*@��������:              ���ڰ�/��ɫ���R����Ļ��͸�ӱ任�������
*@param _src             ����Ҷ�/��ɫͼ��
*@param _dst             �����ʾ���ͻ���ͼ��
*@param border_white     ��׵�ͼ��Ե��������ֵ
*@param border_black     ��ڵ�ͼ��Ե��������ֵ
*@param border_lightleak ��©��ͼ��Ե��������ֵ
*@param Mwhite           �׵�͸�ӱ任����
*@param Mblack           �ڵ�͸�ӱ任����
*@param Mlightleak       ©��͸�ӱ任����
*@param M_white_abshow   ��ʾ�쳣�任����
*@param ID               ��λID��(����)
*@ScreenType_Flag        ��Ļ����
*@�޸�ʱ�䣺		     2020��9��18��
*@��ע˵��              use
=========================================================*/
bool f_MainCam_PersTransMatCal(InputArray _src, int border_white, int border_biankuang, Mat *Mwhite, Mat *Mbiankuang, Mat *M_white_abshow, int ID, String ScreenType_Flag)
{
	//    double screen_long=size_long/size_width;
	//    int screen_long=size_long/size_width;
	bool isArea_1, isArea_2;														//��ʾ�쳣��־λ
	Mat src = _src.getMat();                                                        //����Դͼ��
	if (src.type() == CV_8UC1)														//������8λͼ
		src = src.clone();															//����ԭͼ
	else
		cvtColor(src, src, CV_BGR2GRAY);										    //�ҶȻ���ɫͼ
	CV_Assert(src.depth() == CV_8U);                                                //8λ�޷���
	Mat binaryImage = Mat::zeros(src.size(), CV_8UC1);                              //��ֵͼ��
	threshold(src, binaryImage, 40, 255, CV_THRESH_BINARY);							//��ֵ��(������)
	medianBlur(binaryImage, binaryImage, 5);										//��ֵ�˲�ȥ�����
	int displayError_Areasignal = 0;												//������������ж���ʾ�쳣��־λ
	vector<vector<Point>> contours;													//contours��ŵ㼯��Ϣ
	findContours(binaryImage, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);    //CV_RETR_EXTERNAL����������CV_CHAIN_APPROX_NONE������������Ϣ
	vector<Point> upLinePoint, leftLinePoint, downLinePoint, rightLinePoint;		//�������Ҳ�㼯����
	Vec4f upLine_Fit, leftLine_Fit, downLine_Fit, rightLine_Fit;				    //�����������ֱ������
	vector<Point2f> src_corner(4), src_corner_biankuang(4), src_corner_abshow(4);   //�ĸ����ཻ�õ��ǵ����꣬©��ǵ㣬��ʾ�쳣�ǵ�
	Rect rect;																        //��С����Ӿ���
	int x1, y1, x2, y2, x3, y3, x4, y4;			//���Ӿ����������Ϣ
	for (vector<int>::size_type i = 0; i < contours.size(); i++)
	{
		double area = contourArea(contours[i]);
		if (area > 2500000 && area < 5000000)
		{
			displayError_Areasignal++;
			rect = boundingRect(contours[i]);
			x1 = rect.tl().x;//���Ͻ�
			y1 = rect.tl().y;//���Ͻ�
			x2 = rect.tl().x;//���½�
			y2 = rect.br().y;//���½�
			x3 = rect.br().x;//���½�
			y3 = rect.br().y;//���½�
			x4 = rect.br().x;//���Ͻ�
			y4 = rect.tl().y;//���Ͻ�
			int radianEliminate = 230;
			int deviation = 160;
			for (int j = 0; j < contours[i].size(); j++)
			{
				//���㼯
				if (contours[i][j].y > y1 + radianEliminate && contours[i][j].y < y1 + (y2 - y1)*0.3 && abs(contours[i][j].x - x1) < deviation ||
					contours[i][j].y > y1 + (y2 - y1)*0.7 && contours[i][j].y < y2 - radianEliminate && abs(contours[i][j].x - x1) < deviation)
					leftLinePoint.push_back(Point(contours[i][j].x, contours[i][j].y));
				//�Ҳ�㼯
				if (contours[i][j].y > y1 + radianEliminate && contours[i][j].y < y2 - radianEliminate && abs(contours[i][j].x - x3) < deviation)
					rightLinePoint.push_back(Point(contours[i][j].x, contours[i][j].y));
				//�ϲ�㼯
				if (contours[i][j].x > x1 + radianEliminate && contours[i][j].x < x4 - radianEliminate && abs(contours[i][j].y - y1) < deviation)
					upLinePoint.push_back(Point(contours[i][j].x, contours[i][j].y));
				//�²�㼯
				if (contours[i][j].x > x1 + radianEliminate && contours[i][j].x < x4 - radianEliminate && abs(contours[i][j].y - y2) < deviation)
					downLinePoint.push_back(Point(contours[i][j].x, contours[i][j].y));
			}
		}
	}
	if (leftLinePoint.size() == 0 || rightLinePoint.size() == 0 || upLinePoint.size() == 0 || downLinePoint.size() == 0)
		displayError_Areasignal = 0;
	//������������ж���ʾ�쳣
	if (displayError_Areasignal > 0 && ID == 1)
		isArea_1 = false;
	if (displayError_Areasignal == 0 && ID == 1)
		isArea_1 = true;
	if (displayError_Areasignal > 0 && ID == 2)
		isArea_2 = false;
	if (displayError_Areasignal == 0 && ID == 2)
		isArea_2 = true;
	//δ��ȡ����Ļ�ж���ʾ�쳣��ȡ��Ե����
	if (displayError_Areasignal == 0)
	{
		vector<Point2f> src_points(4);
		src_points = { Point(0, 0), Point(0, 10), Point(10, 10), Point(10, 0) };
		vector<Point2f> dst_points(4);
		if (ScreenType_Flag == "������")
			dst_points = { Point(0, 0), Point(0, 1775), Point(3000, 1775), Point(3000, 0) };
		else        //pixel_num
			//dst_points = { Point(0, 0), Point(0, 1500), Point(3000, 1500), Point(3000, 0) };
			dst_points = { Point(0, 0), Point(0, 1500), Point(3000, 1500), Point(3000, 0) };
		*Mwhite = cv::getPerspectiveTransform(src_points, dst_points);
		//*Mlightleak = cv::getPerspectiveTransform(src_points, dst_points);
		*Mbiankuang = cv::getPerspectiveTransform(src_points, dst_points);
		*M_white_abshow = cv::getPerspectiveTransform(src_points, dst_points);
	}
	//������Ļ��ȡ��Ļ���ĸ��ǵ�
	else
	{
		fitLine(leftLinePoint, leftLine_Fit, cv::DIST_L2, 0, 1e-2, 1e-2);               //������ֱ��
		fitLine(rightLinePoint, rightLine_Fit, cv::DIST_L2, 0, 1e-2, 1e-2);				//�Ҳ����ֱ��
		fitLine(upLinePoint, upLine_Fit, cv::DIST_L2, 0, 1e-2, 1e-2);					//�ϲ����ֱ��
		fitLine(downLinePoint, downLine_Fit, cv::DIST_L2, 0, 1e-2, 1e-2);				//�²����ֱ��

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

		src_corner[0] = getPointSlopeCrossPoint(upLine_Fit, leftLine_Fit);              //���Ͻǵ�
		src_corner[1] = getPointSlopeCrossPoint(downLine_Fit, leftLine_Fit);		    //���½ǵ�
		src_corner[2] = getPointSlopeCrossPoint(downLine_Fit, rightLine_Fit);	        //���½ǵ�
		src_corner[3] = getPointSlopeCrossPoint(upLine_Fit, rightLine_Fit);		        //���Ͻǵ�

																						//��4���ǵ������λ�ý���΢�����׵�ͼ�Լ��ڵ�ͼ��
		src_corner[0].x = src_corner[0].x - border_white;
		src_corner[0].y = src_corner[0].y - border_white;
		src_corner[1].x = src_corner[1].x - border_white;
		src_corner[1].y = src_corner[1].y + border_white;
		src_corner[2].x = src_corner[2].x + border_white;
		src_corner[2].y = src_corner[2].y + border_white;
		src_corner[3].x = src_corner[3].x + border_white;
		src_corner[3].y = src_corner[3].y - border_white;
		//��4���ǵ������λ�ý���΢����©����ͼ��
		src_corner_biankuang[0].x = src_corner[0].x - border_biankuang;
		src_corner_biankuang[0].y = src_corner[0].y - border_biankuang;
		src_corner_biankuang[1].x = src_corner[1].x - border_biankuang;
		src_corner_biankuang[1].y = src_corner[1].y + border_biankuang;
		src_corner_biankuang[2].x = src_corner[2].x + border_biankuang;
		src_corner_biankuang[2].y = src_corner[2].y + border_biankuang;
		src_corner_biankuang[3].x = src_corner[3].x + border_biankuang;
		src_corner_biankuang[3].y = src_corner[3].y - border_biankuang;
		//��ʾ�쳣(�׵�ͼ)
		src_corner_abshow[0].x = src_corner[0].x - border_white + 10;
		src_corner_abshow[0].y = src_corner[0].y - border_white + 10;
		src_corner_abshow[1].x = src_corner[1].x - border_white + 10;
		src_corner_abshow[1].y = src_corner[1].y + border_white - 10;
		src_corner_abshow[2].x = src_corner[2].x + border_white - 10;
		src_corner_abshow[2].y = src_corner[2].y + border_white - 10;
		src_corner_abshow[3].x = src_corner[3].x + border_white - 10;
		src_corner_abshow[3].y = src_corner[3].y - border_white + 10;

		vector<Point2f> dst_corner(4);
		if (ScreenType_Flag == "������")
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
*@�� �� ��:     getPointSlopeCrossPoint
*@��������:     �����бʽ����ֱ�ߵĽ���
*@param LineA   ƽ������
*@param LineB   ��ֱ����
*@����ʱ�䣺    2020��8��17��
*@��ע˵��
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
*@�� �� ��:              f_LeftRightCam_PersTransMatCal
*@��������:              �������R��͸�ӱ任������㺯��
*@param _src             ����Ҷ�/��ɫͼ��
*@param Mwhite           �׵�͸�ӱ任����
*@ScreenType_Flag        ��Ļ����
*@����ʱ�䣺		     2020��8��20��
*@��ע˵��              use
=========================================================*/
bool f_LeftRightCam_PersTransMatCal(InputArray _src, Mat *Mwhite, Mat *M_R_1_E, String ScreenType_Flag, int border_white)
{
	bool Ext_Result_Left_Right;                                                     //��ȡ��Ļ�ɹ���־λ
	Mat src = _src.getMat();                                                        //����Դͼ��
	if (src.type() == CV_8UC1)														//������8λͼ
		src = src.clone();															//����ԭͼ
	else
		cvtColor(src, src, CV_BGR2GRAY);										    //�ҶȻ���ɫͼ
	CV_Assert(src.depth() == CV_8U);                                                //8λ�޷���
	Mat binaryImage = Mat::zeros(src.size(), CV_8UC1);                              //��ֵͼ��
	threshold(src, binaryImage, 40, 255, CV_THRESH_BINARY);							//��ֵ��(������)
	medianBlur(binaryImage, binaryImage, 5);										//��ֵ�˲�ȥ�����
	int displayError_Areasignal = 0;												//������������ж���ʾ�쳣��־λ
	vector<vector<Point>> contours;													//contours��ŵ㼯��Ϣ
	findContours(binaryImage, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);    //CV_RETR_EXTERNAL����������CV_CHAIN_APPROX_NONE������������Ϣ
	vector<Point> upLinePoint, leftLinePoint, downLinePoint, rightLinePoint;		//�������Ҳ�㼯����
	Vec4f upLine_Fit, leftLine_Fit, downLine_Fit, rightLine_Fit;				    //�����������ֱ������
	vector<Point2f> src_corner(4);                                                  //�ĸ����ཻ�õ��ǵ�����
	vector<Point2f> src_corner_enlarge(4);
	Rect rect;																        //��С����Ӿ���
	int x1, y1, x2, y2, x3, y3, x4, y4;			                                    //���Ӿ����������Ϣ
	vector<Point2f> dst_corner(4);                                                  //͸�ӱ任��ĵ����Ϣ
	for (vector<int>::size_type i = 0; i < contours.size(); i++)
	{
		double area = contourArea(contours[i]);
		if (area > 1500000 && area < 5000000)
		{
			displayError_Areasignal++;
			rect = cv::boundingRect(contours[i]);                                           //��С��Ӿ�����ȡ
			Ext_Result_Left_Right = false;                                                             //��ȡ����Ļ

			//cv::rectangle(src, rect, Scalar(255, 0, 0), 5, LINE_8, 0);
			x1 = rect.tl().x;//���Ͻ�
			y1 = rect.tl().y;//���Ͻ�
			x2 = rect.tl().x;//���½�
			y2 = rect.br().y;//���½�
			x3 = rect.br().x;//���½�
			y3 = rect.br().y;//���½�
			x4 = rect.br().x;//���Ͻ�
			y4 = rect.tl().y;//���Ͻ�
																									   //���������С1/3�����õ��µľ��ζ���
			//ȡֱ�ߵĲ�������
			int radianEliminate = 230;//(R��)����ʹ��
			int radianEliminate2 = 360;//(R��)����ʹ��
			int deviation = 200;//(б�ߴ��������)����ʹ��
			int deviation2 = 120;//(б�ߴ��������)����ʹ��

			if (displayError_Areasignal != 0)
			{
				for (int j = 0; j < contours[i].size(); j++)
				{
					//���㼯
					if (contours[i][j].y > y1 + radianEliminate && contours[i][j].y < y1 + (y2 - y1)*0.3 && abs(contours[i][j].x - x1) < deviation ||
						contours[i][j].y > y1 + (y2 - y1)*0.7 && contours[i][j].y < y2 - radianEliminate && abs(contours[i][j].x - x1) < deviation)
						leftLinePoint.push_back(Point(contours[i][j].x, contours[i][j].y));
					//�Ҳ�㼯
					if (contours[i][j].y > y1 + radianEliminate && contours[i][j].y < y2 - radianEliminate && abs(contours[i][j].x - x3) < deviation)
						rightLinePoint.push_back(Point(contours[i][j].x, contours[i][j].y));
					//�ϲ�㼯
					if (contours[i][j].x > x1 + radianEliminate2 && contours[i][j].x < x4 - radianEliminate2 && abs(contours[i][j].y - y1) < deviation2)
						upLinePoint.push_back(Point(contours[i][j].x, contours[i][j].y));
					if (contours[i][j].x > x1 + radianEliminate2 && contours[i][j].x < x4 - radianEliminate2 && abs(contours[i][j].y - y2) < deviation2)
						downLinePoint.push_back(Point(contours[i][j].x, contours[i][j].y));
				}
				//�²�㼯
				//downLinePoint.push_back(Point((x2 + x3) / 2, (y2 + y3) / 2));
				if (leftLinePoint.size() != 0 && rightLinePoint.size() != 0 && upLinePoint.size() != 0 && downLinePoint.size() != 0)
				{
					//ֱ�����
					fitLine(leftLinePoint, leftLine_Fit, cv::DIST_L2, 0, 1e-2, 1e-2);               //������ֱ��
					fitLine(rightLinePoint, rightLine_Fit, cv::DIST_L2, 0, 1e-2, 1e-2);				//�Ҳ����ֱ��
					fitLine(upLinePoint, upLine_Fit, cv::DIST_L2, 0, 1e-2, 1e-2);					//�ϲ����ֱ��
					downLine_Fit[0] = upLine_Fit[0];
					downLine_Fit[1] = upLine_Fit[1];
					downLine_Fit[2] = downLinePoint[0].x;
					downLine_Fit[3] = downLinePoint[0].y;                                           //�²�ֱ��ȷ��

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
					//�ǵ���ȡ
					src_corner[0] = getPointSlopeCrossPoint(upLine_Fit, leftLine_Fit);              //���Ͻǵ�
					src_corner[1] = getPointSlopeCrossPoint(downLine_Fit, leftLine_Fit);		    //���½ǵ�
					src_corner[2] = getPointSlopeCrossPoint(downLine_Fit, rightLine_Fit);	        //���½ǵ�
					src_corner[3] = getPointSlopeCrossPoint(upLine_Fit, rightLine_Fit);		        //���Ͻǵ�


					//src_corner_enlarge[0] = Point2f(xcoordinate1 + tl.x - border_white, ycoordinate1 + tl.y - border_white);	                         //���Ͻ�
					//src_corner_enlarge[1] = Point2f(xcoordinate2 + bl.x - border_white, ycoordinate2 - height / 3 + bl.y + border_white);              //���½�
					//src_corner_enlarge[2] = Point2f(xcoordinate3 - width / 4 + br.x + border_white, ycoordinate3 - height / 3 + br.y + border_white);	 //���½�
					//src_corner_enlarge[3] = Point2f(xcoordinate4 - width / 4 + tr.x + border_white, ycoordinate4 + tr.y - border_white);	             //���Ͻ�

					src_corner_enlarge[0].y = src_corner[0].y - border_white;
					src_corner_enlarge[0].x = src_corner[0].x - border_white;
					src_corner_enlarge[1].y = src_corner[1].y + border_white;
					src_corner_enlarge[1].x = src_corner[1].x - border_white;
					src_corner_enlarge[2].y = src_corner[2].y + border_white;
					src_corner_enlarge[2].x = src_corner[2].x + border_white;
					src_corner_enlarge[3].y = src_corner[3].y - border_white;
					src_corner_enlarge[3].x = src_corner[3].x + border_white;
					//͸�ӱ任�������
					if (ScreenType_Flag == "������")
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
	//û����ȡ����Ļ
	if (displayError_Areasignal == 0)
	{
		Ext_Result_Left_Right = true; //û����ȡ����Ļ
		vector<Point2f> src_points(4);
		src_points = { Point(0, 0), Point(0, 10), Point(10, 10), Point(10, 0) };
		vector<Point2f> dst_points(4);
		if (ScreenType_Flag == "������")
			dst_points = { Point(0, 0), Point(0, 1775), Point(3000, 1775), Point(3000, 0) };
		else
			dst_points = { Point(0, 0), Point(0, 1500), Point(3000, 1500), Point(3000, 0) };
		*Mwhite = cv::getPerspectiveTransform(src_points, dst_points);                        //͸�ӱ任������ȡ
		*M_R_1_E = cv::getPerspectiveTransform(src_points, dst_points);
	}

	return Ext_Result_Left_Right;
}

/*=========================================================
*@�� �� ��:              f_FrontBackCam_PersTransMatCal
*@��������:              ǰ�����R��͸�ӱ任������㺯��
*@param _src             ����Ҷ�/��ɫͼ��
*@param Mwhite           �׵�͸�ӱ任����
*@ScreenType_Flag        ��Ļ����
*@����ʱ�䣺		     2020��8��21��
*@��ע˵��
=========================================================*/
bool f_FrontBackCam_PersTransMatCal(InputArray _src, Mat *Mwhite, String ScreenType_Flag)
{
	bool Ext_Result_Front_Back;                                                     //��ȡ��Ļ�ɹ���־λ
	Mat src = _src.getMat();                                                        //����Դͼ��
	if (src.type() == CV_8UC1)														//������8λͼ
		src = src.clone();															//����ԭͼ
	else
		cvtColor(src, src, CV_BGR2GRAY);										    //�ҶȻ���ɫͼ
	CV_Assert(src.depth() == CV_8U);                                                //8λ�޷���
	Mat binaryImage = Mat::zeros(src.size(), CV_8UC1);                              //��ֵͼ��
	threshold(src, binaryImage, 40, 255, CV_THRESH_BINARY);							//��ֵ��(������)
	medianBlur(binaryImage, binaryImage, 5);										//��ֵ�˲�ȥ�����
	int displayError_Areasignal = 0;												//������������ж���ʾ�쳣��־λ
	vector<vector<Point>> contours;													//contours��ŵ㼯��Ϣ
	findContours(binaryImage, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);    //CV_RETR_EXTERNAL����������CV_CHAIN_APPROX_NONE������������Ϣ
	vector<Point> upLinePoint, leftLinePoint, downLinePoint, rightLinePoint;		//�������Ҳ�㼯����
	Vec4f upLine_Fit, leftLine_Fit, downLine_Fit, rightLine_Fit;				    //�����������ֱ������
	vector<Point2f> src_corner(4);                                                  //�ĸ����ཻ�õ��ǵ�����
	Rect rect;																        //��С����Ӿ���
	int x1, y1, x2, y2, x3, y3, x4, y4;			                                    //���Ӿ����������Ϣ
	vector<Point2f> dst_corner(4);                                                  //͸�ӱ任��ĵ����Ϣ
	for (vector<int>::size_type i = 0; i < contours.size(); i++)
	{
		double area = contourArea(contours[i]);
		if (area > 2000000 && area < 5000000)
		{
			displayError_Areasignal++;
			rect = cv::boundingRect(contours[i]);                                           //��С��Ӿ�����ȡ
			Ext_Result_Front_Back = false;                                                             //��ȡ����Ļ
			//���������С1/3�����õ��µľ��ζ���
			int PixelGap1 = rect.tl().x;
			int PixelGap2 = src.cols - (rect.tl().x + rect.width);
			//�������ȡ
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
			//�������ȡ
			y1 = rect.tl().y;
			y2 = rect.br().y;
			if (y2 >= src.rows)
				y2 = src.rows - 1;
			y3 = y2;
			y4 = y1;
			//ȡֱ�ߵĲ�������
			int radianEliminate = 350;//(R��)����ʹ��
			int radianEliminate1 = 480;//(R��)����ʹ��
			int radianEliminate2 = 230;//(R��)����ʹ��
			int deviation = 120;//(б�ߴ��������)����ʹ��
			int deviation2 = 200;//(б�ߴ��������)����ʹ��
			if (PixelGap1 > PixelGap2)
			{
				//��Ӿ�����С
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
						//���㼯
						if (contours[i][j].y > y1 + radianEliminate && contours[i][j].y < y2 - radianEliminate1 && abs(contours[i][j].x - x1) < deviation)
							leftLinePoint.push_back(Point(contours[i][j].x, contours[i][j].y));
						//�ϲ�㼯
						if (contours[i][j].x > x1 + radianEliminate2 && contours[i][j].x < x4 - radianEliminate2 && abs(contours[i][j].y - y1) < deviation2)
							upLinePoint.push_back(Point(contours[i][j].x, contours[i][j].y));
						//�²�㼯
						if (contours[i][j].x > x1 + radianEliminate2 && contours[i][j].x < x4 - radianEliminate2 && abs(contours[i][j].y - y2) < deviation2)
							downLinePoint.push_back(Point(contours[i][j].x, contours[i][j].y));
					}
					//�Ҳ�㼯
					rightLinePoint.push_back(Point((x3 + x4) / 2, (y3 + y4) / 2));
					if (leftLinePoint.size() != 0 && upLinePoint.size() != 0 && downLinePoint.size() != 0)
					{
						//ֱ�����
						fitLine(leftLinePoint, leftLine_Fit, cv::DIST_L2, 0, 1e-2, 1e-2);               //������ֱ��
						rightLine_Fit[0] = leftLine_Fit[0];
						rightLine_Fit[1] = leftLine_Fit[1];
						rightLine_Fit[2] = rightLinePoint[0].x;
						rightLine_Fit[3] = rightLinePoint[0].y;                                         //�Ҳ����ֱ��
						fitLine(upLinePoint, upLine_Fit, cv::DIST_L2, 0, 1e-2, 1e-2);					//�ϲ����ֱ��
						fitLine(downLinePoint, downLine_Fit, cv::DIST_L2, 0, 1e-2, 1e-2);               //�²����ֱ��
						//�ǵ���ȡ
						src_corner[0] = getPointSlopeCrossPoint(upLine_Fit, leftLine_Fit);              //���Ͻǵ�
						src_corner[1] = getPointSlopeCrossPoint(downLine_Fit, leftLine_Fit);		    //���½ǵ�
						src_corner[2] = getPointSlopeCrossPoint(downLine_Fit, rightLine_Fit);	        //���½ǵ�
						src_corner[3] = getPointSlopeCrossPoint(upLine_Fit, rightLine_Fit);		        //���Ͻǵ�
						//͸�ӱ任�������
						if (ScreenType_Flag == "������")
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
				//��Ӿ�����С
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
						//�Ҳ�㼯
						if (contours[i][j].y > y1 + radianEliminate1 && contours[i][j].y < y1 + (y2 - y1) * 0.3 &&abs(contours[i][j].x - x3) < deviation || contours[i][j].y > y1 + (y2 - y1)*0.7 && contours[i][j].y < y2 - radianEliminate && abs(contours[i][j].x - x3) < deviation)
							rightLinePoint.push_back(Point(contours[i][j].x, contours[i][j].y));
						//�ϲ�㼯
						if (contours[i][j].x > x1 + radianEliminate2 && contours[i][j].x < x4 - radianEliminate2 && abs(contours[i][j].y - y1) < deviation2)
							upLinePoint.push_back(Point(contours[i][j].x, contours[i][j].y));
						//�²�㼯
						if (contours[i][j].x > x1 + radianEliminate2 && contours[i][j].x < x4 - radianEliminate2 && abs(contours[i][j].y - y2) < deviation2)
							downLinePoint.push_back(Point(contours[i][j].x, contours[i][j].y));
					}
					//���㼯
					leftLinePoint.push_back(Point((x1 + x2) / 2, (y1 + y2) / 2));
					if (rightLinePoint.size() != 0 && upLinePoint.size() != 0 && downLinePoint.size() != 0)
					{
						//���ֱ��
						fitLine(rightLinePoint, rightLine_Fit, cv::DIST_L2, 0, 1e-2, 1e-2);				//�Ҳ����ֱ��
						leftLine_Fit[0] = rightLine_Fit[0];
						leftLine_Fit[1] = rightLine_Fit[1];
						leftLine_Fit[2] = leftLinePoint[0].x;
						leftLine_Fit[3] = leftLinePoint[0].y;                                           //������ֱ��
						fitLine(upLinePoint, upLine_Fit, cv::DIST_L2, 0, 1e-2, 1e-2);				    //�ϲ����ֱ��
						fitLine(downLinePoint, downLine_Fit, cv::DIST_L2, 0, 1e-2, 1e-2);				//�²����ֱ��
						//�ǵ���ȡ
						src_corner[0] = getPointSlopeCrossPoint(upLine_Fit, leftLine_Fit);              //���Ͻǵ�
						src_corner[1] = getPointSlopeCrossPoint(downLine_Fit, leftLine_Fit);		    //���½ǵ�
						src_corner[2] = getPointSlopeCrossPoint(downLine_Fit, rightLine_Fit);	        //���½ǵ�
						src_corner[3] = getPointSlopeCrossPoint(upLine_Fit, rightLine_Fit);		        //���Ͻǵ�
						//͸�ӱ任�������
						if (ScreenType_Flag == "������")
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
	//û����ȡ����Ļ
	if (displayError_Areasignal == 0)
	{
		Ext_Result_Front_Back = true; //û����ȡ����Ļ
		vector<Point2f> src_points(4);
		src_points = { Point(0, 0), Point(0, 10), Point(10, 10), Point(10, 0) };
		vector<Point2f> dst_points(4);
		if (ScreenType_Flag == "������")
			dst_points = { Point(0, 0), Point(0, 1775), Point(3000, 1775), Point(3000, 0) };
		else
			dst_points = { Point(0, 0), Point(0, 1500), Point(3000, 1500), Point(3000, 0) };
		*Mwhite = cv::getPerspectiveTransform(src_points, dst_points);                        //͸�ӱ任������ȡ
	}

	return Ext_Result_Front_Back;
}




