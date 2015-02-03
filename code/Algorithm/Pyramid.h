#pragma once
#include "../Header.h"
#include "morph.h"

typedef struct
{
public:
	int w;
	int h;
	float inverse_wh;
	int blocks_row;//how many tiles in a row
	int blocks_col;//how many tiles in a row
	int blocks_num;//how many tiles in a row
	cv::Mat image1;//image1
	cv::Mat image2;//image2
	cv::Mat mask1;//mask1
	cv::Mat mask2;//mask2
}CPyramid;

class CPyramids
{
public:
	QVector<CPyramid> levels;
	cv::Mat _vector;
	cv::Mat _qpath;
	cv::Mat _error;
	cv::Mat _extends1;
	cv::Mat _extends2;
	int _order;
	QPoint _p1;
	QPoint _p2;
	Morph *_gpu;
	CPyramids(void);
	~CPyramids(void);
	void build_pyramid(cv::Mat &image1, cv::Mat &image2, Parameters& para, int n, int order,bool gpu_flag);
	void build_pyramid(cv::Mat& mask1, cv::Mat& mask2);

};
