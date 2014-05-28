#pragma once
#include "../Header.h"

class CQuadraticPath: public QThread
{
	Q_OBJECT

public:
	CQuadraticPath(int layer_index,cv::Mat& vector, cv::Mat& qpath,bool gpu_flag);
	~CQuadraticPath(void);
	void run();
	void run_cuda();
	void run_cpu();
	void run_level(cv::Mat& _vector, cv::Mat& _qpath);
	
public:
signals:
	void sigFinished(int index);

public:
	float _runtime;
	cv::Mat& _qpath;
	cv::Mat& _vector;
	int w,h;
	int _levels;
	int _layer_index;
	bool _gpu_flag;

};
