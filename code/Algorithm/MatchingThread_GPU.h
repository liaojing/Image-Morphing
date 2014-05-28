#pragma once

#include "..//Header.h"
#include "Pyramid.h"
#include "imgio.h"
#include "morph.h"

class CMatchingThread_GPU: public QThread
{
	Q_OBJECT
public:
signals:
	void sigUpdate(int index);
	void sigFinished(int index);

public slots:
		void update_result();

public:
	void run();
	static bool cb(const std::string &text, int cur, int max,const rod::dimage<float2> *halfway,const rod::dimage<float> *ssim_error,void *data);
	CMatchingThread_GPU(Parameters parameters,CPyramids &pyramids, int layer_index);
	~CMatchingThread_GPU(void);

public:
	CPyramids& _pyramids;
	Parameters _parameters;
	int _layer_index;
	static int _total_iter,_current_iter;
	float run_time;
	QTimer *_timer;
	static bool runflag;
	static std::vector<float2> _halfway;
	static std::vector<float> _error;	
};

