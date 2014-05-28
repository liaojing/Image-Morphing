#include "MatchingThread_GPU.h"
bool CMatchingThread_GPU::runflag;
int CMatchingThread_GPU::_current_iter;
int CMatchingThread_GPU::_total_iter;
std::vector<float2> CMatchingThread_GPU::_halfway;
std::vector<float> CMatchingThread_GPU::_error;	


CMatchingThread_GPU::CMatchingThread_GPU(Parameters parameters,CPyramids &pyramids, int layer_index):_pyramids(pyramids),_layer_index(layer_index)

{
	runflag=true;	
	_total_iter=_current_iter=0;	

	_timer=NULL;
	_timer=new QTimer(this);
	connect(_timer,SIGNAL(timeout()), this, SLOT(update_result()) );	
	_timer->start(1000);

    _pyramids._gpu->set_callback(&cb);
}


CMatchingThread_GPU::~CMatchingThread_GPU(void)
{
}

void CMatchingThread_GPU::run()
{      
 	clock_t start, finish;
 	start = clock();
	 rod::dimage<float2> halfway;

 	if(_pyramids._gpu->calculate_halfway_parametrization(halfway))
 	{
			_timer->stop();
			finish = clock(); 
 			run_time = (float)(finish - start) / CLOCKS_PER_SEC*1000; 
			halfway.copy_to_host(_halfway);
			update_result();
 			emit sigFinished(_layer_index);//finished signal
 	}
}


bool CMatchingThread_GPU::cb(const std::string &text, int cur, int max,
              const rod::dimage<float2> *halfway,
              const rod::dimage<float> *ssim_error,
              void *data)
{
	_current_iter=cur;
	_total_iter=max;
    if(halfway)
    {		
		halfway->copy_to_host(_halfway);
 		//ssim_error->copy_to_host(_error);
 	 }     
	
    return runflag;
}

void CMatchingThread_GPU::update_result()
{

 if(_halfway.size()>0)
 { 
	#pragma omp parallel for
 	for (int y=0;y<_pyramids._vector.rows;y++)
 		for(int x=0;x<_pyramids._vector.cols;x++)
 		{
 			int index=y*_pyramids._vector.cols+x;
 			_pyramids._vector.at<Vec3f>(y,x)=Vec3f(_halfway[index].x,_halfway[index].y,0);
 			//_pyramids._error.at<float>(y,x)=_error[index];
 		}		
	}
 emit sigUpdate(_layer_index);	

}

