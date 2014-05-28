#pragma once
#include "../Header.h"
class CPoissonExt: public QThread
{
	Q_OBJECT
public:
	CPoissonExt(int layer_index,cv::Mat& vector,cv::Mat& extends1, cv::Mat& extends2,bool gpu_flag);
	void run();
	~CPoissonExt(void);
	int prepare(int side, cv::Mat &extends);
	void poissonExtend(cv::Mat &dst, int size);
	void poissonExtend_cuda(cv::Mat &dst,int size);
	//void Result_mesh();
	template<class T_in, class T_out>
	inline T_out BilineaGetColor_clamp(cv::Mat& img, float px,float py);//clamp for outside of the boundary

public:
signals:
	void sigFinished(int index);

public:
	int w,h,ex;
	int *type,*index;
	cv::Mat _image1,_image2;
	cv::Mat &_extends1,&_extends2;
	cv::Mat &_vector;
	float _runtime;
	int _layer_index;
	bool _gpu_flag;
};

