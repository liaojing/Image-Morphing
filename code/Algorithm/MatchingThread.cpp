#include "MatchingThread.h"

#define p2(x) (x*x)
#define SIGN(n) (n>0?1:(n<0?-1:0))

class TTW; class TW; class TT; class T;
class BBW; class BW; class BB; class B;
class RRW; class RW; class RR; class R;
class LLW; class LW; class LL; class L;

template <class CTT,class CT,class CBB,class CB,
          class CLL,class CL,class CRR,class CR>
class Nei;

template<int ytype, int xtype>
class Oper;

CMatchingThread::CMatchingThread(Parameters& parameters,CPyramids &pyramids, int layer_index):_pyramids(pyramids),_parameters(parameters),_layer_index(layer_index)
{
	runflag=true;
	_total_l=_pyramids.levels.count();
	_current_l=_total_l-1;

	_total_iter=_current_iter=0;
	int iter_num=_parameters.max_iter;
	for (int el=_total_l-2;el>=0;el--)
	{
		iter_num/=_parameters.max_iter_drop_factor;
		_total_iter+=iter_num*_pyramids.levels[el].w*_pyramids.levels[el].h;
	}

	data=(CData*)malloc(_total_l*sizeof(CData));
	for(int el=0;el<_total_l;el++)
	{
		data[el].ssim.luma0=NULL;
		data[el].improving_mask=NULL;
		data[el].vx=NULL;
		data[el].vy=NULL;
	}

	_timer=NULL;
 	_timer=new QTimer(this);
    connect(_timer,SIGNAL(timeout()), this, SLOT(update_result()) );
	_timer->start(1000);

}

CMatchingThread::~CMatchingThread()
{
	if(_timer)
		delete _timer;

	for(int el=0;el<_total_l;el++)
	{
		//all inremental
		if(data[el].ssim.luma0)
			free(data[el].ssim.luma0);

		//improving mask
		if(data[el].improving_mask)
			free(data[el].improving_mask);

		//vector
		if(data[el].vx)
			free(data[el].vx);
		if(data[el].vy)
			free(data[el].vy);
	}
	free(data);


}

void CMatchingThread::update_result()
{
	if(_current_l<0) _current_l=0;
	if(data[_current_l].vx&&data[_current_l].vy&&data[_current_l].ssim.luma0&&runflag)
	{
		cv::Mat result_c(_pyramids.levels[_current_l].h,_pyramids.levels[_current_l].w,CV_32FC3);
		cv::Mat error_c(_pyramids.levels[_current_l].h,_pyramids.levels[_current_l].w,CV_32FC1);

		float ratio_x,ratio_y;
		ratio_x=(float)_pyramids.levels[0].w/(float)_pyramids.levels[_current_l].w;
		ratio_y=(float)_pyramids.levels[0].h/(float)_pyramids.levels[_current_l].h;

		#pragma omp parallel for
		for(int y=0;y<_pyramids.levels[_current_l].h;y++)
			for(int x=0;x<_pyramids.levels[_current_l].w;x++)
			{
				int index=mem_index(x,y,_current_l);
				float vx=data[_current_l].vx[index]*ratio_x;
				float vy=data[_current_l].vy[index]*ratio_y;
				result_c.at<Vec3f>(y,x)=Vec3f(vx,vy,0.0f);
				error_c.at<float>(y,x)=data[_current_l].ssim.value[index];
			}

		cv::resize(result_c,_pyramids._vector,Size(_pyramids._vector.cols,_pyramids._vector.rows),0,0,INTER_LINEAR);
		cv::resize(error_c,_pyramids._error,Size(_pyramids._error.cols,_pyramids._error.rows),0,0,INTER_LINEAR);
		emit sigUpdate(_layer_index);
	}
}

void CMatchingThread::run()
{
	//time
	clock_t start, finish;

	load_identity(_total_l-1);
	start = clock();
	if (_parameters.ui_points.size()>0)
		optimize_highestlevel(_total_l-1);


	_max_iter=_parameters.max_iter;
	for (_current_l = _total_l-2; _current_l >= 0; _current_l--) {
		_max_iter/=_parameters.max_iter_drop_factor;
		upsample_level(_current_l+1, _current_l);
		initialize_incremental(_current_l);
		start = clock();
		optimize_level(_current_l);

		if(!runflag)
			{
				break;
			}
	}
	finish = clock();
	run_time = (float)(finish - start)*1000 / CLOCKS_PER_SEC;

	_timer->stop();
	update_result();
	emit sigFinished(_layer_index);//finished signal
}




//void CMatchingThread::upsample_level(int c_el, int n_el)
//{
//	int size=_pyramids.levels[n_el].blocks_num*25*sizeof(float);
//	data[n_el].vx=(float*)malloc(size);
//	data[n_el].vy=(float*)malloc(size);
//	memset(data[n_el].vx,0,size);
//	memset(data[n_el].vy,0,size);
//
//	#pragma omp parallel for
//	for(int y=0;y<_pyramids.levels[n_el].h;y++)
//		for(int x=0;x<_pyramids.levels[n_el].w;x++)
//		{
//			int index=mem_index(x,y,n_el);		
//			float xx=(x+0.5)/2-0.5;
//			float yy=(y+0.5)/2-0.5;
//
//			int near_x=xx+0.5;
//			int near_y=yy+0.5;
//			int far_x=near_x+SIGN(xx-near_x);
//			int far_y=near_y+SIGN(yy-near_y);
//
//			if (near_x<0) near_x=0;
//			if (near_y<0) near_y=0;
//			if (near_x>_pyramids.levels[c_el].w-1) near_x=_pyramids.levels[c_el].w-1;
//			if (near_y>_pyramids.levels[c_el].h-1) near_y=_pyramids.levels[c_el].h-1;
//			if (far_x<0) far_x=0;
//			if (far_y<0) far_y=0;
//			if (far_x>_pyramids.levels[c_el].w-1) far_x=_pyramids.levels[c_el].w-1;
//			if (far_y>_pyramids.levels[c_el].h-1) far_y=_pyramids.levels[c_el].h-1;
//
//			int nn=mem_index(near_x,near_y,c_el);
//			int nf=mem_index(near_x,far_y,c_el);
//			int fn=mem_index(far_x,near_y,c_el);
//			int ff=mem_index(far_x,far_y,c_el);
//
//			data[n_el].vx[index]=2.0f*(data[c_el].vx[nn]*0.5625+data[c_el].vx[nf]*0.1875+data[c_el].vx[fn]*0.1875+data[c_el].vx[ff]*0.0625);
//			data[n_el].vy[index]=2.0f*(data[c_el].vy[nn]*0.5625+data[c_el].vy[nf]*0.1875+data[c_el].vy[fn]*0.1875+data[c_el].vy[ff]*0.0625);
//		}
//}


 void CMatchingThread::upsample_level(int c_el, int n_el)
 {
 	float ratio_x,ratio_y;
 	ratio_x=(float)_pyramids.levels[n_el].w/(float)_pyramids.levels[c_el].w;
 	ratio_y=(float)_pyramids.levels[n_el].h/(float)_pyramids.levels[c_el].h;
 	cv::Mat result_c(_pyramids.levels[c_el].h,_pyramids.levels[c_el].w,CV_32FC2);
 	cv::Mat result_n(_pyramids.levels[n_el].h,_pyramids.levels[n_el].w,CV_32FC2);
 
 #pragma omp parallel for
 	for(int y=0;y<_pyramids.levels[c_el].h;y++)
 		for(int x=0;x<_pyramids.levels[c_el].w;x++)
 		{
 			int index=mem_index(x,y,c_el);
 			float vx=data[c_el].vx[index]*ratio_x;
 			float vy=data[c_el].vy[index]*ratio_y;
 			result_c.at<Vec2f>(y,x)=Vec2f(vx,vy);
 		}
 
 		cv::resize(result_c,result_n,Size(_pyramids.levels[n_el].w,_pyramids.levels[n_el].h),0,0,INTER_LINEAR);
 
 		int size=_pyramids.levels[n_el].blocks_num*25*sizeof(float);
 		data[n_el].vx=(float*)malloc(size);
 		data[n_el].vy=(float*)malloc(size);
 		memset(data[n_el].vx,0,size);
 		memset(data[n_el].vy,0,size);
 
 #pragma omp parallel for
 		for(int y=0;y<_pyramids.levels[n_el].h;y++)
 			for(int x=0;x<_pyramids.levels[n_el].w;x++)
 			{
 				int index=mem_index(x,y,n_el);
 				Vec2f v=result_n.at<Vec2f>(y,x);
 				data[n_el].vx[index]=v[0];
 				data[n_el].vy[index]=v[1];
 			}
 }

void CMatchingThread::load_identity(int el)
{
	int size=_pyramids.levels[el].blocks_num*25*sizeof(float);
	data[el].vx=(float*)malloc(size);
	data[el].vy=(float*)malloc(size);
	memset(data[el].vx,0,size);
	memset(data[el].vy,0,size);

}

void CMatchingThread::initialize_incremental(int el)
{

	int size=_pyramids.levels[el].blocks_num*25;
	float* incremental_pointer=(float*)malloc(size*16*sizeof(float));
	memset(incremental_pointer,0,size*16*sizeof(float));

	//ssim
	data[el].ssim.luma0=incremental_pointer;
	data[el].ssim.luma1=incremental_pointer+size*1;
	data[el].ssim.mean0=incremental_pointer+size*2;
	data[el].ssim.mean1=incremental_pointer+size*3;
	data[el].ssim.var0=incremental_pointer+size*4;
	data[el].ssim.var1=incremental_pointer+size*5;
	data[el].ssim.cross01=incremental_pointer+size*6;
	data[el].ssim.value=incremental_pointer+size*7;
	data[el].ssim.counter=incremental_pointer+size*8;

	//tps
	data[el].tps.axy=incremental_pointer+size*9;
	data[el].tps.bx=incremental_pointer+size*10;
	data[el].tps.by=incremental_pointer+size*11;


	//ui
	data[el].ui.axy=incremental_pointer+size*12;
	data[el].ui.bx=incremental_pointer+size*13;
	data[el].ui.by=incremental_pointer+size*14;

	//mask
	data[el].mask_ign=incremental_pointer+size*15;

	//improving
	data[el].improving=1;
	data[el].improving_mask=(bool*)malloc(_pyramids.levels[el].blocks_num*25*sizeof(bool));

	//initialize for ui
	for(size_t i=0;i<_parameters.ui_points.size();i++)
	{
		float x0=_parameters.ui_points[i].lp.x*_pyramids.levels[el].w-0.5f;
		float y0=_parameters.ui_points[i].lp.y*_pyramids.levels[el].h-0.5f;
		float x1=_parameters.ui_points[i].rp.x*_pyramids.levels[el].w-0.5f;
		float y1=_parameters.ui_points[i].rp.y*_pyramids.levels[el].h-0.5f;
		float con_x=(x0+x1)/2.0f;
		float con_y=(y0+y1)/2.0f;
		float vx=(x1-x0)/2.0f;
		float vy=(y1-y0)/2.0f;

		for(int y=floor(con_y);y<=ceil(con_y);y++)
			for(int x=floor(con_x);x<=ceil(con_x);x++)
			{
				if(inside(x,y,el))
				{
					int index=mem_index(x,y,el);

					float bilinear_w=(1.0-fabs(y-con_y))*(1.0-fabs(x-con_x));

					data[el].ui.axy[index]+=bilinear_w;
					data[el].ui.bx[index]+=2.0f*bilinear_w*(data[el].vx[index]-vx);
					data[el].ui.by[index]+=2.0f*bilinear_w*(data[el].vy[index]-vy);
				}
			}
	}

	//initialize each pixel for tps and ssim
	outerloop<Init>(el);
}



void CMatchingThread::optimize_level(int el) {

	int iter=0;
	while (data[el].improving>0&&iter<_max_iter) {

		data[el].improving=0;
		outerloop<Opt>(el);
		iter++;
		_current_iter+=_pyramids.levels[el].w*_pyramids.levels[el].h;
		if(!runflag)
			break;
	}

	iter_num[el]=iter;
	_current_iter+=(_max_iter-iter)*_pyramids.levels[el].w*_pyramids.levels[el].h;

}

void CMatchingThread::optimize_highestlevel(int el)//svd
{
	int w=_pyramids.levels[el].w;
	int h=_pyramids.levels[el].h;
	int num=w*h;
	CvMat*   A   =   cvCreateMat(num,num,CV_32FC1);
	CvMat*   Bx  =   cvCreateMat(num,1,CV_32FC1);
	CvMat*   By  =   cvCreateMat(num,1,CV_32FC1);
	CvMat*   X  =   cvCreateMat(num,1,CV_32FC1);
	CvMat*   Y  =   cvCreateMat(num,1,CV_32FC1);
	cvZero(A);
	cvZero(Bx);
	cvZero(By);
	cvZero(X);
	cvZero(Y);

	//set matrixs for tps
	int step = A->step/sizeof (float );
	float *A_data = A->data.fl;
	#pragma omp parallel for
	for(int y=0;y<h;y++)
		for(int x=0;x<w;x++)
		{
			int i=y*w+x;
			//dxx
			if(x>1)
				(A_data+i*step)[i-2]+=1.0f,	(A_data+i*step)[i-1]+=-2.0f,	(A_data+i*step)[i]+=1.0f;
			if(x>0&&x<w-1)
				(A_data+i*step)[i-1]+=-2.0f, (A_data+i*step)[i]+=4.0f,	(A_data+i*step)[i+1]+=-2.0f;
			if(x<w-2)
				(A_data+i*step)[i]+=1.0f,	(A_data+i*step)[i+1]+=-2.0f,	(A_data+i*step)[i+2]+=1.0f;
			//dy
			if(y>1)
				(A_data+i*step)[i-2*w]+=1.0f, (A_data+i*step)[i-w]+=-2.0f,	(A_data+i*step)[i]+=1.0f;
			if(y>0&&y<h-1)
				(A_data+i*step)[i-w]+=-2.0f,  (A_data+i*step)[i]+=4.0f,		(A_data+i*step)[i+w]+=-2.0f;
			if(y<h-2)
				(A_data+i*step)[i]+=1.0f,	 (A_data+i*step)[i+w]+=-2.0f,	(A_data+i*step)[i+2*w]+=1.0f;

			//dxy
			if(x>0&&y>0)
				(A_data+i*step)[i-w-1]+=2.0f,	(A_data+i*step)[i-w]+=-2.0f,	(A_data+i*step)[i-1]+=-2.0f,	(A_data+i*step)[i]+=2.0f;
			if(x<w-1&&y>0)
				(A_data+i*step)[i-w]+=-2.0f,	(A_data+i*step)[i-w+1]+=2.0f,	(A_data+i*step)[i]+=2.0f,	(A_data+i*step)[i+1]+=-2.0f;
			if(x>0&&y<h-1)
				(A_data+i*step)[i-1]+=-2.0f,	(A_data+i*step)[i]+=2.0f,	(A_data+i*step)[i+w-1]+=2.0f,	(A_data+i*step)[i+w]+=-2.0f;
			if(x<w-1&&y<h-1)
				(A_data+i*step)[i]+=2.0f,	(A_data+i*step)[i+1]+=-2.0f,	(A_data+i*step)[i+w]+=-2.0f,	(A_data+i*step)[i+w+1]+=2.0f;

		}

		//set matrix for ui
		for(size_t i=0;i<_parameters.ui_points.size();i++)
		{
			float x0=_parameters.ui_points[i].lp.x*_pyramids.levels[el].w-0.5f;
			float y0=_parameters.ui_points[i].lp.y*_pyramids.levels[el].h-0.5f;
			float x1=_parameters.ui_points[i].rp.x*_pyramids.levels[el].w-0.5f;
			float y1=_parameters.ui_points[i].rp.y*_pyramids.levels[el].h-0.5f;
			float con_x=(x0+x1)/2.0f;
			float con_y=(y0+y1)/2.0f;
			float vx=(x1-x0)/2.0f;
			float vy=(y1-y0)/2.0f;

			for(int y=floor(con_y);y<=ceil(con_y);y++)
				for(int x=floor(con_x);x<=ceil(con_x);x++)
				{
					if(inside(x,y,el))
					{
						float bilinear_w=(1.0-fabs(y-con_y))*(1.0-fabs(x-con_x));
						int i=y*w+x;
						cvmSet(A,i,i,cvmGet(A,i,i)+bilinear_w);
						cvmSet(Bx,i,0,cvmGet(Bx,i,0)+bilinear_w*vx);
						cvmSet(By,i,0,cvmGet(By,i,0)+bilinear_w*vy);
					}
				}

		}

		//set boundary condistion
		int x,y,i;
		switch(_parameters.bcond)
		{
		case BCOND_NONE:
			break;

		case BCOND_CORNER://corner
			x=0,y=0;
			i=y*w+x;
			(A_data+i*step)[i]+=10.f;

			x=0,y=h-1;
			i=y*w+x;
			(A_data+i*step)[i]+=10.f;

			x=w-1,y=h-1;
			i=y*w+x;
			(A_data+i*step)[i]+=10.f;

			x=w-1,y=0;
			i=y*w+x;
			(A_data+i*step)[i]+=10.f;
			break;

		case BCOND_BORDER:
			for (x=0;x<w;x++)
			{
				y=0;
				i=y*w+x;
				(A_data+i*step)[i]+=10.f;

				y=h-1;
				i=y*w+x;
				(A_data+i*step)[i]+=10.f;
			}

			for(y=1;y<h-1;y++)
			{
				x=0;
				i=y*w+x;
				(A_data+i*step)[i]+=10.f;

				x=w-1;
				i=y*w+x;
				(A_data+i*step)[i]+=10.f;

			}

			break;
		}

		if(!cvSolve(A,   Bx,   X,   CV_LU))
			cvSolve(A,   Bx,   X,   CV_SVD);
		if(!cvSolve(A,   By,   Y,   CV_LU))
			cvSolve(A,   By,   Y,   CV_SVD);

		//load to vx,vy
		#pragma omp parallel for
		for(int y=0;y<h;y++)
		{
			for(int x=0;x<w;x++)
			{
				int index=mem_index(x,y,el);

				data[el].vx[index]=cvmGet(X,y*w+x,0);
				data[el].vy[index]=cvmGet(Y,y*w+x,0);
			}
		}

		cvReleaseMat(&A);
		cvReleaseMat(&Bx);
		cvReleaseMat(&By);
		cvReleaseMat(&X);
		cvReleaseMat(&Y);

	}




template <class FUN>
void CMatchingThread::outerloop(int el)
{
	loop_inside_ny<TTW,	TW,	BB,	B,	FUN>(el,0);
	loop_inside_ny<TTW,	T,	BB,	B,	FUN>(el,1);
	loop_inside_ny<TT,	T,	BB,	B,	FUN>(el,2);
	loop_inside_ny<TT,	T,	BBW,B,	FUN>(el,3);
	loop_inside_ny<TT,	T,	BBW,BW,	FUN>(el,4);
}

template <class CTT,class CT, class CBB, class CB,class FUN>
void CMatchingThread::loop_inside_ny(int el,int ny)
{
	loop_inside_nx<CTT, CT, CBB, CB, LLW, LW,RR,  R, FUN>(el,ny,0);
	loop_inside_nx<CTT, CT, CBB, CB, LLW, L, RR,  R, FUN>(el,ny,1);
	loop_inside_nx<CTT, CT, CBB, CB, LL,  L, RR,  R, FUN>(el,ny,2);
	loop_inside_nx<CTT, CT, CBB, CB, LL,  L, RRW, R, FUN>(el,ny,3);
	loop_inside_nx<CTT, CT, CBB, CB, LL,  L, RRW, RW,FUN>(el,ny,4);

}

template <class CTT,class CT, class CBB, class CB,class CLL, class CL, class CRR, class CR,class FUN>
void CMatchingThread::loop_inside_nx(int el,int ny,int nx)
{
	int y=ny;
	for(;y<1;y+=5)
		loop_inside_by<0,Nei<CTT,CT,CBB,CB,CLL,CL,CRR,CR>,FUN>(el,nx,y);

	for(;y<2;y+=5)
		loop_inside_by<1,Nei<CTT,CT,CBB,CB,CLL,CL,CRR,CR>,FUN>(el,nx,y);

	int num=0;
#pragma omp parallel for
	for(int i=y;i<_pyramids.levels[el].h-2;i+=5)
	{
		loop_inside_by<2,Nei<CTT,CT,CBB,CB,CLL,CL,CRR,CR>,FUN>(el,nx,i);
#pragma omp atomic
		num++;
	}
#pragma omp barrier

	for(y+=num*5;y<_pyramids.levels[el].h-1;y+=5)
		loop_inside_by<3,Nei<CTT,CT,CBB,CB,CLL,CL,CRR,CR>,FUN>(el,nx,y);

	for(;y<_pyramids.levels[el].h;y+=5)
		loop_inside_by<4,Nei<CTT,CT,CBB,CB,CLL,CL,CRR,CR>,FUN>(el,nx,y);
}

template <int ytype, class CNEI,class FUN>
void CMatchingThread::loop_inside_by(int el, int nx,int y)
{
	int x=nx;
	for(;x<1;x+=5)
		loop_inside_bx<ytype,0,CNEI,FUN>(el,y,x);

	for(;x<2;x+=5)
		loop_inside_bx<ytype,1,CNEI,FUN>(el,y,x);

	for(;x<_pyramids.levels[el].w-2;x+=5)
		loop_inside_bx<ytype,2,CNEI,FUN>(el,y,x);

	for(;x<_pyramids.levels[el].w-1;x+=5)
		loop_inside_bx<ytype,3,CNEI,FUN>(el,y,x);

	for(;x<_pyramids.levels[el].w;x+=5)
		loop_inside_bx<ytype,4,CNEI,FUN>(el,y,x);
}

template<int ytype,int xtype, class CNEI,class FUN>
void CMatchingThread::loop_inside_bx(int el,int y, int x)
{
	CNEI nei(_pyramids.levels[el].blocks_row,_pyramids.levels[el].blocks_col);
	FUN fun;
	Oper<ytype,xtype> op;

	Pixel p;
	p.x=x;p.y=y;
	p.index=mem_index(x,y,el);

	fun.run(el,p,nei,op,this);

};

//Gradient descent
template <class CNEI, class COPER>
void Opt::run(int el,Pixel& p,CNEI& nei,COPER& op,CMatchingThread* match)
{
	int w=match->_pyramids.levels[el].w;
	int h=match->_pyramids.levels[el].h;
	if (!op.io.flag(p.index,nei,match->data[el].improving_mask))
		return;
	match->data[el].improving_mask[p.index]=false;

	float gx, gy;
	match->compute_gradient(el,p,nei,op,gx,gy);
	float ng = sqrt(p2(gx) + p2(gy));
	if(ng==0.0f)
		return;
	gx/=ng;
	gy/=ng;

	//different boundary condition
	switch(match->_parameters.bcond)
	{
	case BCOND_NONE:
		break;
	case BCOND_CORNER:
		if((p.x==0&&p.y==0)||(p.x==0&&p.y==h-1)||(p.x==w-1&&p.y==0)||(p.x==w-1&&p.y==h-1))
			return;
		break;
	case BCOND_BORDER:
		if(p.x==0||p.y==0||p.y==h-1||p.x==w-1)
			return;
		break;
                /*
	case BCOND_RECT:
		if(p.x==0||p.x==w-1)
			gx=0;
		if(p.y==0||p.y==h-1)
			gy=0;
		break;
                */
	}


	//golden section
	float t0 = 0.0f;
	float t3=match->prevent_foldover(el,p,nei,op, gx, gy);
	const float R = 0.618033989f;
	const float C= 1.0f-R;
	float t1 = R*t0+C*t3;
	float t2 = R*t1+C*t3;
	float f1 = match->energy_change(el,p,nei,op,gx*t1, gy*t1);
	float f2 =match->energy_change(el,p,nei,op,gx*t2, gy*t2);

	while (t3 - t0 > match->_parameters.eps) {
		if(f2<f1)
		{
			t0=t1;
			t1=t2;
			t2=R*t2+C*t3;
			f1=f2;
			f2=match->energy_change(el,p,nei,op,gx*t2, gy*t2);
		}
		else
		{
			t3=t2;
			t2=t1;
			t1=R*t1+C*t0;
			f2=f1;
			f1=match->energy_change(el,p,nei,op,gx*t1, gy*t1);
		}

	}

	float tmin,fmin;
	if(f1<f2)
		tmin=t1,fmin=f1;
	else
		tmin=t2,fmin=f2;


	// commit changes?
	float dx = gx*tmin, dy = gy*tmin;
	if (fmin < 0.f) 	{
		match->commit_pixel_motion(el,p,nei,op,dx, dy);
		match->data[el].improving++;
	}
}



template <class CNEI, class COPER>
void Init:: run(int el, Pixel& p,CNEI& nei,COPER& op,CMatchingThread* match)
{
	NBMatrix vx;//Motion vector
	NBMatrix vy;//Motion vector

	op.io.readNB(p.index,nei,vx,match->data[el].vx);
	op.io.readNB(p.index,nei,vy,match->data[el].vy);

	//ssim
	NBMatrix luma0,luma1;
	NBMatrix mask;
	for(int y=0;y<5;y++)
		for(int x=0;x<5;x++)
		{
			luma0.data[y][x]=match->BilineaGetColor_clamp(match->_pyramids.levels[el].image1,p.x+x-2-vx.data[y][x],p.y+y-2-vy.data[y][x]);
			luma1.data[y][x]=match->BilineaGetColor_clamp(match->_pyramids.levels[el].image2,p.x+x-2+vx.data[y][x],p.y+y-2+vy.data[y][x]);
			float mask0=match->BilineaGetColor_fill(match->_pyramids.levels[el].mask1,p.x+x-2-vx.data[y][x],p.y+y-2-vy.data[y][x],255.0f);
			float mask1=match->BilineaGetColor_fill(match->_pyramids.levels[el].mask2,p.x+x-2+vx.data[y][x],p.y+y-2+vy.data[y][x],255.0f);
			mask.data[y][x]=match->ModMask_ign(mask0,mask1);
		}

		match->data[el].ssim.luma0[p.index]=luma0.data[2][2];
		match->data[el].ssim.luma1[p.index]=luma1.data[2][2];
		match->data[el].mask_ign[p.index]=mask.data[2][2];

		float counter=match->data[el].ssim.counter[p.index]=op.io.sumNB(mask);
		float m0=match->data[el].ssim.mean0[p.index]=op.io.sumNB(luma0*mask);
		float m1=match->data[el].ssim.mean1[p.index]=op.io.sumNB(luma1*mask);

		NBMatrix var0,var1;
		var0=luma0*luma0;
		var1=luma1*luma1;
		float v0=match->data[el].ssim.var0[p.index]=op.io.sumNB(var0*mask);
		float v1=match->data[el].ssim.var1[p.index]=op.io.sumNB(var1*mask);

		NBMatrix cross01;
		cross01=luma0*luma1;
		float cr01=match->data[el].ssim.cross01[p.index]=op.io.sumNB(cross01*mask);

		match->data[el].ssim.value[p.index]=match->ssim(m0,m1,v0,v1,cr01,counter);

		//TPS
		match->data[el].tps.axy[p.index]=op.tps.stencil[2][2]/2;
		match->data[el].tps.bx[p.index]=op.tps.get(p.index,nei,match->data[el].vx);
		match->data[el].tps.by[p.index]=op.tps.get(p.index,nei,match->data[el].vy);

		//improving map
		match->data[el].improving_mask[p.index]=true;//allow moving
}

template <class CNEI,class COPER>
void CMatchingThread::compute_gradient(int el,Pixel& p,CNEI& nei,COPER& op, float &gx, float &gy) {

	gx=-(energy_change(el,p,nei,op,_parameters.eps,0.0f)-energy_change(el,p,nei,op,-_parameters.eps,0.0f));
	gy=-(energy_change(el,p,nei,op,0.0f,_parameters.eps)-energy_change(el,p,nei,op,0.0f,-_parameters.eps));

}

template <class CNEI,class COPER>
float CMatchingThread::prevent_foldover(int el, Pixel& p,CNEI& nei,COPER& op, float gx, float gy)
{
	//image
	float cx0,cy0,cx1,cy1;
	cx0=p.x-data[el].vx[p.index];
	cy0=p.y-data[el].vy[p.index];
	cx1=p.x+data[el].vx[p.index];
	cy1=p.y+data[el].vy[p.index];

	float nx0[8],ny0[8],nx1[8],ny1[8];
	op.io.oneringNB(p,nei,nx0,ny0,nx1,ny1,data[el].vx,data[el].vy);

	float td[16], d[16];
	int inter_num=0;
	for(int i=0;i<8;i++)
	{
		if(intersect(cx0, cy0, cx0-gx, cy0-gy, nx0[i], ny0[i], nx0[(i+1)%8], ny0[(i+1)%8], td[inter_num], d[inter_num]))
			inter_num++;
		if(intersect(cx1, cy1, cx1+gx, cy1+gy, nx1[i], ny1[i], nx1[(i+1)%8], ny1[(i+1)%8], td[inter_num], d[inter_num]))
			inter_num++;
	}

	//find the smallest non-negative t
	if(inter_num==0)
		return 1.0f;

	float td_min=1.0f,d_min=0.0f;
	for(int i=0;i<inter_num;i++)
	{
		if (td[i]>=0&&td[i]*d_min<d[i]*td_min)
			td_min=td[i],d_min=d[i];
	}

	if(fabs(d_min)>0.00001f)
		return MAX(td_min/d_min-_parameters.eps,0.0f);
	else
		return 1.0f;
}



int CMatchingThread::intersect(float x1, float y1, float x2, float y2,float x3, float y3, float x4, float y4, float &td, float &d)
{
	d = (y4-y3)*(x2-x1)-(x4-x3)*(y2-y1);
	td =(x4-x3)*(y1-y3)-(y4-y3)*(x1-x3);
	float ud =(x2-x1)*(y1-y3)-(y2-y1)*(x1-x3);

	if (d > 0) {
		if (ud >= 0&& ud <= d)
			return 1;
		else
			return 0;
	}
	else {
		if (ud <= 0 && ud >= d) {
			td = -td;
			d = -d;
			return 1;
		}
		else
			return 0;
	}
}


template <class CNEI,class COPER>
float CMatchingThread::energy_change(int el,Pixel& p,CNEI& nei,COPER& op,float dx, float dy)//dx,dy is the moving vector
{
	float vx=data[el].vx[p.index];
	float vy=data[el].vy[p.index];

	//ign_mask
	float old_mask,new_mask;
	old_mask=data[el].mask_ign[p.index];
	float mask1=BilineaGetColor_fill(_pyramids.levels[el].mask1,p.x-vx-dx,p.y-vy-dy,255.0f);
	float mask2=BilineaGetColor_fill(_pyramids.levels[el].mask2,p.x+vx+dx,p.y+vy+dy,255.0f);
	new_mask=ModMask_ign(mask1,mask2);

	//ssim
	NBMatrix mean0,mean1,var0,var1,cross01,counter,new_ssim,old_ssim;
	op.io.readNB(p.index,nei,old_ssim,data[el].ssim.value);
	float luma0,luma1;
	ssim_update(mean0, mean1,var0,var1,cross01,counter,new_ssim,luma0,luma1,el,p,nei,op,vx+dx, vy+dy,old_mask,new_mask);
	float v_ssim=op.io.sumNB(new_ssim-old_ssim);

	//tps
	float v_tps=0.0f;
	v_tps+=data[el].tps.axy[p.index]*p2(dx);
	v_tps+=data[el].tps.axy[p.index]*p2(dy);
	v_tps+=data[el].tps.bx[p.index]*dx;
	v_tps+=data[el].tps.by[p.index]*dy;


	//ui
	float v_ui=0.0f;
	v_ui+=data[el].ui.axy[p.index]*p2(dx);
	v_ui+=data[el].ui.axy[p.index]*p2(dy);
	v_ui+=data[el].ui.bx[p.index]*dx;
	v_ui+=data[el].ui.by[p.index]*dy;

	return (_parameters.w_ui*v_ui+_parameters.w_ssim*v_ssim)*_pyramids.levels[el].inverse_wh+_parameters.w_tps*v_tps;

}

template <class CNEI,class COPER>
void CMatchingThread::commit_pixel_motion(int el, Pixel& p,CNEI &nei, COPER &op, float dx, float dy)
{
	float vx=data[el].vx[p.index];
	float vy=data[el].vy[p.index];

	//ign_mask
	float old_mask,new_mask;
	old_mask=data[el].mask_ign[p.index];
	float mask1=BilineaGetColor_fill(_pyramids.levels[el].mask1,p.x-vx-dx,p.y-vy-dy,255.0f);
	float mask2=BilineaGetColor_fill(_pyramids.levels[el].mask2,p.x+vx+dx,p.y+vy+dy,255.0f);
	new_mask=data[el].mask_ign[p.index]=ModMask_ign(mask1,mask2);

	//ssim
	NBMatrix mean0,mean1,var0,var1,cross01,counter,value;
	float luma0,luma1;
	ssim_update(mean0, mean1,var0,var1,cross01,counter,value,luma0,luma1,el,p,nei,op,vx+dx, vy+dy,old_mask,new_mask);
	op.io.writeNB(p.index,nei,mean0,data[el].ssim.mean0);
	op.io.writeNB(p.index,nei,mean1,data[el].ssim.mean1);
	op.io.writeNB(p.index,nei,var0,data[el].ssim.var0);
	op.io.writeNB(p.index,nei,var1,data[el].ssim.var1);
	op.io.writeNB(p.index,nei,cross01,data[el].ssim.cross01);
	op.io.writeNB(p.index,nei,counter,data[el].ssim.counter);
	op.io.writeNB(p.index,nei,value,data[el].ssim.value);
	data[el].ssim.luma0[p.index]=luma0;
	data[el].ssim.luma1[p.index]=luma1;

	//tps
	op.tps.update(p.index,nei,dx,data[el].tps.bx);
	op.tps.update(p.index,nei,dy,data[el].tps.by);

	//ui
	data[el].ui.bx[p.index]+=2.0f*dx*data[el].ui.axy[p.index];
	data[el].ui.by[p.index]+=2.0f*dy*data[el].ui.axy[p.index];

	//vector
	data[el].vx[p.index]+=dx;
	data[el].vy[p.index]+=dy;

	//imprving mask
	data[el].improving_mask[p.index]=true;

}


float CMatchingThread::ssim(float mean0,float mean1,float var0, float var1, float cross01, float counter)
{
	if(counter<=1.0f) return 0.0f;
	//_parameters
	const float c2=58.5225f;
	const float c3=29.26125f;

	mean0/=counter;
	mean1/=counter;

	var0=(var0-counter*mean0*mean0)/counter;
	var0=MAX(0.0f,var0);

	var1=(var1-counter*mean1*mean1)/counter;
	var1=MAX(0.0f,var1);

	cross01=(cross01-counter*mean0*mean1)/counter;

	float var0_root=sqrt(var0);
	float var1_root=sqrt(var1);
	float c=(2*var0_root*var1_root+c2)/(var0+var1+c2);
	float s=(fabs(cross01)+c3)/(var0_root*var1_root+c3);

	return MIN(MAX(0.0f,1.0f-c*s),1.0-_parameters.ssim_clamp);
}

template <class CNEI,class COPER>
void CMatchingThread::ssim_update(NBMatrix& mean0, NBMatrix& mean1,NBMatrix& var0,NBMatrix& var1,NBMatrix& cross01,NBMatrix& counter, NBMatrix& value,float& luma0,float& luma1,int el,Pixel& p,CNEI& nei,COPER& op, float vx, float vy,float old_mask, float new_mask)
{
	//ssim
	float old_luma0=data[el].ssim.luma0[p.index];
	float old_luma1=data[el].ssim.luma1[p.index];
	luma0=BilineaGetColor_clamp(_pyramids.levels[el].image1,p.x-vx,p.y-vy);
	luma1=BilineaGetColor_clamp(_pyramids.levels[el].image2,p.x+vx,p.y+vy);


	op.io.readNB(p.index,nei,mean0,data[el].ssim.mean0);
	op.io.readNB(p.index,nei,mean1,data[el].ssim.mean1);
	op.io.readNB(p.index,nei,var0,data[el].ssim.var0);
	op.io.readNB(p.index,nei,var1,data[el].ssim.var1);
	op.io.readNB(p.index,nei,cross01,data[el].ssim.cross01);
	op.io.readNB(p.index,nei,counter,data[el].ssim.counter);

	mean0=mean0+(luma0*new_mask-old_luma0*old_mask);
	mean1=mean1+(luma1*new_mask-old_luma1*old_mask);
	var0=var0+(p2(luma0)*new_mask-p2(old_luma0)*old_mask);
	var1=var1+(p2(luma1)*new_mask-p2(old_luma1)*old_mask);
	cross01=cross01+(luma0*luma1*new_mask-old_luma0*old_luma1*old_mask);
	counter=counter+(new_mask-old_mask);

	for(int y=0;y<5;y++)
		for(int x=0;x<5;x++)
		{
			value.data[y][x]=ssim(mean0.data[y][x],mean1.data[y][x],var0.data[y][x],var1.data[y][x],cross01.data[y][x],counter.data[y][x]);
		}

}

////inline functions////
float CMatchingThread::BilineaGetColor_clamp(cv::Mat& img, float px,float py)//clamp for outside of the boundary
{
	int x[2],y[2];
	float value[2][2];
	int w=img.cols;
	int h=img.rows;
	x[0]=floor(px);
	y[0]=floor(py);
	x[1]=ceil(px);
	y[1]=ceil(py);

	float u=px-x[0];
	float v=py-y[0];

	for (int i=0;i<2;i++)
		for(int j=0;j<2;j++)
		{
			int temp_x,temp_y;
			temp_x=x[i];
			temp_y=y[j];
			temp_x=MAX(0,temp_x);
			temp_x=MIN(w-1,temp_x);
			temp_y=MAX(0,temp_y);
			temp_y=MIN(h-1,temp_y);
			value[i][j]=img.at<uchar>(temp_y,temp_x);
		}


		return
			value[0][0]*(1-u)*(1-v)+value[0][1]*(1-u)*v+value[1][0]*u*(1-v)+value[1][1]*u*v;
}

float CMatchingThread::BilineaGetColor_fill(cv::Mat& img, float px,float py,float fill)//clamp for outside of the boundary
{
	int x[2],y[2];
	float value[2][2];
	int w=img.cols;
	int h=img.rows;
	x[0]=floor(px);
	y[0]=floor(py);
	x[1]=ceil(px);
	y[1]=ceil(py);

	float u=px-x[0];
	float v=py-y[0];

	for (int i=0;i<2;i++)
		for(int j=0;j<2;j++)
		{
			int temp_x,temp_y;
			temp_x=x[i];
			temp_y=y[j];
			if(temp_x<0||temp_x>w-1||temp_y<0||temp_y>h-1)
				value[i][j]=fill;
			else
				value[i][j]=img.at<uchar>(temp_y,temp_x);
		}


		return
			value[0][0]*(1-u)*(1-v)+value[0][1]*(1-u)*v+value[1][0]*u*(1-v)+value[1][1]*u*v;
}
inline float CMatchingThread::ModMask_ign(float mask1,float mask2)
{

	return (1.0f-mask1/255.0)*(1.0f-mask2/255.0);

}


inline bool CMatchingThread::inside(float x, float y,int el)
{
	if (y>=0.0f&&y<_pyramids.levels[el].h&&x>=0.0f&&x<_pyramids.levels[el].w)
		return true;
	else
		return false;
}

inline int CMatchingThread::mem_index(int x,int y,int el) {
	int color_index=y%5*5+x%5;
	int block_index=y/5*_pyramids.levels[el].blocks_row+x/5;
	return color_index*_pyramids.levels[el].blocks_num+block_index;
}

//class as function

//get neighbor
class TT
{
public:
	TT(int row,int col)
	{
		_offset=-row*col*10;
	}

	int operator()(int index) const {
		return index + _offset;
	}


private:
	int _offset;
}
;

class TTW
{
public:
	TTW(int row,int col)
	{
		_offset=(row*col)*15-row;
	}

	int operator()(int index) const {
		return index + _offset;
	}

private:
	int _offset;
};

class T
{
public:
	T(int row,int col)
	{
		_offset=-row*col*5;
	}

	int operator()(int index) const {
		return index + _offset;
	}

private:
	int _offset;
};


class TW
{
public:
	TW(int row,int col)
	{
		_offset=row*col*20-row;
	}

	int operator()(int index) const {
		return index + _offset;
	}

private:
	int _offset;
};



class BB
{
public:
	BB(int row,int col)
	{
		_offset=row*col*10;
	}

	int operator()(int index) const {
		return index + _offset;
	}

private:
	int _offset;
};


class BBW
{
public:
	BBW(int row,int col)
	{
		_offset=-row*col*15+row;
	}

	int operator()(int index) const {
		return index + _offset;
	}

private:
	int _offset;
};


class B
{
public:
	B(int row,int col)
	{
		_offset=row*col*5;
	}

	int operator()(int index) const {
		return index + _offset;
	}

private:
	int _offset;
};


class BW
{
public:
	BW(int row,int col)
	{
		_offset=-row*col*20+row;
	}

	int operator()(int index) const {
		return index + _offset;
	}

private:
	int _offset;
};


class LL
{
public:
	LL(int row,int col)
	{
		_offset=-2*row*col;
	}

	int operator()(int index) const {
		return index + _offset;
	}

private:
	int _offset;
};


class LLW
{
public:
	LLW(int row,int col)
	{
		_offset=row*col*3-1;
	}

	int operator()(int index) const {
		return index + _offset;
	}

private:
	int _offset;
};


class L
{
public:
	L(int row,int col)
	{
		_offset=-row*col;
	}

	int operator()(int index) const {
		return index + _offset;
	}

private:
	int _offset;
};



class LW
{
public:
	LW(int row,int col)
	{
		_offset=row*col*4-1;
	}

	int operator()(int index) const {
		return index + _offset;
	}

private:
	int _offset;
};



class RR
{
public:
	RR(int row,int col)
	{
		_offset=row*col*2;
	}

	int operator()(int index) const {
		return index + _offset;
	}

private:
	int _offset;
};


class RRW
{
public:
	RRW(int row,int col)
	{
		_offset=-row*col*3+1;
	}

	int operator()(int index) const {
		return index + _offset;
	}

private:
	int _offset;
};



class R
{
public:
	R(int row,int col)
	{
		_offset=row*col;
	}

	int operator()(int index) const {
		return index + _offset;
	}

private:
	int _offset;
};


class RW
{
public:
	RW(int row,int col)
	{
		_offset=-row*col*4+1;
	}

	int operator()(int index) const {
		return index + _offset;
	}

private:
	int _offset;
};


template <class CTT,class CT,class CBB,class CB,class CLL,class CL,class CRR,class CR>
class Nei
{
public:
	CTT tt;
	CT t;
	CBB bb;
	CB b;
	CLL ll;
	CL l;
	CRR rr;
	CR r;

	Nei(int row, int col):tt(row,col),t(row,col),bb(row,col),b(row,col),ll(row,col),l(row,col),rr(row,col),r(row,col)
	{
	}
};



template<int ytype, int xtype>
class IO
{
public:
	static const int stencil[5][5];
	static const int counter;

	template <class CNEI>
	void readNB(int index,CNEI &nei,NBMatrix &matrix,float* src_data){
		if(stencil[0][0]!=0)	matrix.data[0][0]=src_data[nei.ll(nei.tt(index))];
		if(stencil[0][1]!=0)	matrix.data[0][1]=src_data[nei.l(nei.tt(index))];
		if(stencil[0][2]!=0)	matrix.data[0][2]=src_data[nei.tt(index)];
		if(stencil[0][3]!=0)	matrix.data[0][3]=src_data[nei.r(nei.tt(index))];
		if(stencil[0][4]!=0)	matrix.data[0][4]=src_data[nei.rr(nei.tt(index))];
		if(stencil[1][0]!=0)	matrix.data[1][0]=src_data[nei.ll(nei.t(index))];
		if(stencil[1][1]!=0)	matrix.data[1][1]=src_data[nei.l(nei.t(index))];
		if(stencil[1][2]!=0)	matrix.data[1][2]=src_data[nei.t(index)];
		if(stencil[1][3]!=0)	matrix.data[1][3]=src_data[nei.r(nei.t(index))];
		if(stencil[1][4]!=0)	matrix.data[1][4]=src_data[nei.rr(nei.t(index))];
		if(stencil[2][0]!=0)	matrix.data[2][0]=src_data[nei.ll(index)];
		if(stencil[2][1]!=0)	matrix.data[2][1]=src_data[nei.l(index)];
		if(stencil[2][2]!=0)	matrix.data[2][2]=src_data[index];
		if(stencil[2][3]!=0)	matrix.data[2][3]=src_data[nei.r(index)];
		if(stencil[2][4]!=0)	matrix.data[2][4]=src_data[nei.rr(index)];
		if(stencil[3][0]!=0)	matrix.data[3][0]=src_data[nei.ll(nei.b(index))];
		if(stencil[3][1]!=0)	matrix.data[3][1]=src_data[nei.l(nei.b(index))];
		if(stencil[3][2]!=0)	matrix.data[3][2]=src_data[nei.b(index)];
		if(stencil[3][3]!=0)	matrix.data[3][3]=src_data[nei.r(nei.b(index))];
		if(stencil[3][4]!=0)	matrix.data[3][4]=src_data[nei.rr(nei.b(index))];
		if(stencil[4][0]!=0)	matrix.data[4][0]=src_data[nei.ll(nei.bb(index))];
		if(stencil[4][1]!=0)	matrix.data[4][1]=src_data[nei.l(nei.bb(index))];
		if(stencil[4][2]!=0)	matrix.data[4][2]=src_data[nei.bb(index)];
		if(stencil[4][3]!=0)	matrix.data[4][3]=src_data[nei.r(nei.bb(index))];
		if(stencil[4][4]!=0)	matrix.data[4][4]=src_data[nei.rr(nei.bb(index))];
	}

	template <class CNEI>
	void writeNB(int index,CNEI &nei,NBMatrix &matrix,float* dst_data)
	{
		if(stencil[0][0]!=0)	dst_data[nei.ll(nei.tt(index))]=matrix.data[0][0];
		if(stencil[0][1]!=0)	dst_data[nei.l(nei.tt(index))]=matrix.data[0][1];
		if(stencil[0][2]!=0)	dst_data[nei.tt(index)]=matrix.data[0][2];
		if(stencil[0][3]!=0)	dst_data[nei.r(nei.tt(index))]=matrix.data[0][3];
		if(stencil[0][4]!=0)	dst_data[nei.rr(nei.tt(index))]=matrix.data[0][4];
		if(stencil[1][0]!=0)	dst_data[nei.ll(nei.t(index))]=matrix.data[1][0];
		if(stencil[1][1]!=0)	dst_data[nei.l(nei.t(index))]=matrix.data[1][1];
		if(stencil[1][2]!=0)	dst_data[nei.t(index)]=matrix.data[1][2];
		if(stencil[1][3]!=0)	dst_data[nei.r(nei.t(index))]=matrix.data[1][3];
		if(stencil[1][4]!=0)	dst_data[nei.rr(nei.t(index))]=matrix.data[1][4];
		if(stencil[2][0]!=0)	dst_data[nei.ll(index)]=matrix.data[2][0];
		if(stencil[2][1]!=0)	dst_data[nei.l(index)]=matrix.data[2][1];
		if(stencil[2][2]!=0)	dst_data[index]=matrix.data[2][2];
		if(stencil[2][3]!=0)	dst_data[nei.r(index)]=matrix.data[2][3];
		if(stencil[2][4]!=0)	dst_data[nei.rr(index)]=matrix.data[2][4];
		if(stencil[3][0]!=0)	dst_data[nei.ll(nei.b(index))]=matrix.data[3][0];
		if(stencil[3][1]!=0)	dst_data[nei.l(nei.b(index))]=matrix.data[3][1];
		if(stencil[3][2]!=0)	dst_data[nei.b(index)]=matrix.data[3][2];
		if(stencil[3][3]!=0)	dst_data[nei.r(nei.b(index))]=matrix.data[3][3];
		if(stencil[3][4]!=0)	dst_data[nei.rr(nei.b(index))]=matrix.data[3][4];
		if(stencil[4][0]!=0)	dst_data[nei.ll(nei.bb(index))]=matrix.data[4][0];
		if(stencil[4][1]!=0)	dst_data[nei.l(nei.bb(index))]=matrix.data[4][1];
		if(stencil[4][2]!=0)	dst_data[nei.bb(index)]=matrix.data[4][2];
		if(stencil[4][3]!=0)	dst_data[nei.r(nei.bb(index))]=matrix.data[4][3];
		if(stencil[4][4]!=0)	dst_data[nei.rr(nei.bb(index))]=matrix.data[4][4];
	}



	float sumNB(const NBMatrix& matrix)
	{
		float sum=0.0f;
		if(stencil[0][0]!=0)	sum+=matrix.data[0][0];
		if(stencil[0][1]!=0)	sum+=matrix.data[0][1];
		if(stencil[0][2]!=0)	sum+=matrix.data[0][2];
		if(stencil[0][3]!=0)	sum+=matrix.data[0][3];
		if(stencil[0][4]!=0)	sum+=matrix.data[0][4];
		if(stencil[1][0]!=0)	sum+=matrix.data[1][0];
		if(stencil[1][1]!=0)	sum+=matrix.data[1][1];
		if(stencil[1][2]!=0)	sum+=matrix.data[1][2];
		if(stencil[1][3]!=0)	sum+=matrix.data[1][3];
		if(stencil[1][4]!=0)	sum+=matrix.data[1][4];
		if(stencil[2][0]!=0)	sum+=matrix.data[2][0];
		if(stencil[2][1]!=0)	sum+=matrix.data[2][1];
		if(stencil[2][2]!=0)	sum+=matrix.data[2][2];
		if(stencil[2][3]!=0)	sum+=matrix.data[2][3];
		if(stencil[2][4]!=0)	sum+=matrix.data[2][4];
		if(stencil[3][0]!=0)	sum+=matrix.data[3][0];
		if(stencil[3][1]!=0)	sum+=matrix.data[3][1];
		if(stencil[3][2]!=0)	sum+=matrix.data[3][2];
		if(stencil[3][3]!=0)	sum+=matrix.data[3][3];
		if(stencil[3][4]!=0)	sum+=matrix.data[3][4];
		if(stencil[4][0]!=0)	sum+=matrix.data[4][0];
		if(stencil[4][1]!=0)	sum+=matrix.data[4][1];
		if(stencil[4][2]!=0)	sum+=matrix.data[4][2];
		if(stencil[4][3]!=0)	sum+=matrix.data[4][3];
		if(stencil[4][4]!=0)	sum+=matrix.data[4][4];
		return sum;
	}

	template <class CNEI>
	bool flag(int index,CNEI nei,bool *src_data)
	{
		if(stencil[0][0]!=0)	{if(src_data[nei.ll(nei.tt(index))]) return true;}
		if(stencil[0][1]!=0)	{if(src_data[nei.l(nei.tt(index))]) return true;}
		if(stencil[0][2]!=0)	{if(src_data[nei.tt(index)]) return true;}
		if(stencil[0][3]!=0)	{if(src_data[nei.r(nei.tt(index))]) return true;}
		if(stencil[0][4]!=0)	{if(src_data[nei.rr(nei.tt(index))]) return true;}
		if(stencil[1][0]!=0)	{if(src_data[nei.ll(nei.t(index))]) return true;}
		if(stencil[1][1]!=0)	{if(src_data[nei.l(nei.t(index))]) return true;}
		if(stencil[1][2]!=0)	{if(src_data[nei.t(index)]) return true;}
		if(stencil[1][3]!=0)	{if(src_data[nei.r(nei.t(index))]) return true;}
		if(stencil[1][4]!=0)	{if(src_data[nei.rr(nei.t(index))]) return true;}
		if(stencil[2][0]!=0)	{if(src_data[nei.ll(index)]) return true;}
		if(stencil[2][1]!=0)	{if(src_data[nei.l(index)]) return true;}
		if(stencil[2][2]!=0)	{if(src_data[index]) return true;}
		if(stencil[2][3]!=0)	{if(src_data[nei.r(index)]) return true;}
		if(stencil[2][4]!=0)	{if(src_data[nei.rr(index)]) return true;}
		if(stencil[3][0]!=0)	{if(src_data[nei.ll(nei.b(index))]) return true;}
		if(stencil[3][1]!=0)	{if(src_data[nei.l(nei.b(index))]) return true;}
		if(stencil[3][2]!=0)	{if(src_data[nei.b(index)]) return true;}
		if(stencil[3][3]!=0)	{if(src_data[nei.r(nei.b(index))]) return true;}
		if(stencil[3][4]!=0)	{if(src_data[nei.rr(nei.b(index))]) return true;}
		if(stencil[4][0]!=0)	{if(src_data[nei.ll(nei.bb(index))]) return true;}
		if(stencil[4][1]!=0)	{if(src_data[nei.l(nei.bb(index))]) return true;}
		if(stencil[4][2]!=0)	{if(src_data[nei.bb(index)]) return true;}
		if(stencil[4][3]!=0)	{if(src_data[nei.r(nei.bb(index))]) return true;}
		if(stencil[4][4]!=0)	{if(src_data[nei.rr(nei.bb(index))]) return true;}

		return false;
	}
	template <class CNEI>
	void oneringNB(Pixel &p,CNEI &nei,  float nx0[8],float ny0[8],float nx1[8],float ny1[8], float *vx, float* vy)//in clockwise order
	{
		int index=p.index;
		float cx0,cy0,cx1,cy1;
		cx0=p.x-vx[index];
		cy0=p.y-vy[index];
		cx1=p.x+vx[index];
		cy1=p.y+vy[index];

		if(stencil[1][1]!=0)
		{
			nx0[0]=p.x-1.0f-vx[nei.l(nei.t(index))];
			ny0[0]=p.y-1.0f-vy[nei.l(nei.t(index))];
			nx1[0]=p.x-1.0f+vx[nei.l(nei.t(index))];
			ny1[0]=p.y-1.0f+vy[nei.l(nei.t(index))];
		}
		else
		{
			nx0[0]=cx0-1.0f;
			ny0[0]=cy0-1.0f;
			nx1[0]=cx1-1.0f;
			ny1[0]=cy1-1.0f;
		}

		if (stencil[1][2]!=0)
		{
			nx0[1]=p.x-vx[nei.t(index)];
			ny0[1]=p.y-1.0f-vy[nei.t(index)];
			nx1[1]=p.x+vx[nei.t(index)];
			ny1[1]=p.y-1.0f+vy[nei.t(index)];
		}
		else
		{
			nx0[1]=cx0;
			ny0[1]=cy0-1.0f;
			nx1[1]=cx1;
			ny1[1]=cy1-1.0f;
		}

		if (stencil[1][3]!=0)
		{
			nx0[2]=p.x+1.0f-vx[nei.r(nei.t(index))];
			ny0[2]=p.y-1.0f-vy[nei.r(nei.t(index))];
			nx1[2]=p.x+1.0f+vx[nei.r(nei.t(index))];
			ny1[2]=p.y-1.0f+vy[nei.r(nei.t(index))];
		}
		else
		{
			nx0[2]=cx0+1.0f;
			ny0[2]=cy0-1.0f;
			nx1[2]=cx1+1.0f;
			ny1[2]=cy1-1.0f;
		}


		if(stencil[2][3]!=0)
		{
			nx0[3]=p.x+1.0f-vx[nei.r(index)];
			ny0[3]=p.y-vy[nei.r(index)];
			nx1[3]=p.x+1.0f+vx[nei.r(index)];
			ny1[3]=p.y+vy[nei.r(index)];
		}
		else
		{
			nx0[3]=cx0+1.0f;
			ny0[3]=cy0;
			nx1[3]=cx1+1.0f;
			ny1[3]=cy1;
		}


		if(stencil[3][3]!=0)
		{
			nx0[4]=p.x+1.0f-vx[nei.r(nei.b(index))];
			ny0[4]=p.y+1.0f-vy[nei.r(nei.b(index))];
			nx1[4]=p.x+1.0f+vx[nei.r(nei.b(index))];
			ny1[4]=p.y+1.0f+vy[nei.r(nei.b(index))];
		}
		else
		{
			nx0[4]=cx0+1.0f;
			ny0[4]=cy0+1.0f;
			nx1[4]=cx1+1.0f;
			ny1[4]=cy1+1.0f;
		}

		if(stencil[3][2]!=0)
		{
			nx0[5]=p.x-vx[nei.b(index)];
			ny0[5]=p.y+1.0f-vy[nei.b(index)];
			nx1[5]=p.x+vx[nei.b(index)];
			ny1[5]=p.y+1.0f+vy[nei.b(index)];
		}
		else
		{
			nx0[5]=cx0;
			ny0[5]=cy0+1.0f;
			nx1[5]=cx1;
			ny1[5]=cy1+1.0f;
		}

		if(stencil[3][1]!=0)
		{
			nx0[6]=p.x-1.0f-vx[nei.l(nei.b(index))];
			ny0[6]=p.y+1.0f-vy[nei.l(nei.b(index))];
			nx1[6]=p.x-1.0f+vx[nei.l(nei.b(index))];
			ny1[6]=p.y+1.0f+vy[nei.l(nei.b(index))];
		}
		else
		{
			nx0[6]=cx0-1.0f;
			ny0[6]=cy0+1.0f;
			nx1[6]=cx1-1.0f;
			ny1[6]=cy1+1.0f;
		}

		if(stencil[2][1]!=0)
		{
			nx0[7]=p.x-1.0f-vx[nei.l(index)];
			ny0[7]=p.y-vy[nei.l(index)];
			nx1[7]=p.x-1.0f+vx[nei.l(index)];
			ny1[7]=p.y+vy[nei.l(index)];
		}
		else
		{
			nx0[7]=cx0-1.0f;
			ny0[7]=cy0;
			nx1[7]=cx1-1.0f;
			ny1[7]=cy1;
		}
	}
};

template<> const int IO<0,0>::stencil[5][5]={{0,0,0,0,0},{0,0,0,0,0},{0,0,1,1,1},{0,0,1,1,1},{0,0,1,1,1}};
template<> const int IO<0,1>::stencil[5][5]={{0,0,0,0,0},{0,0,0,0,0},{0,1,1,1,1},{0,1,1,1,1},{0,1,1,1,1}};
template<> const int IO<0,2>::stencil[5][5]={{0,0,0,0,0},{0,0,0,0,0},{1,1,1,1,1},{1,1,1,1,1},{1,1,1,1,1}};
template<> const int IO<0,3>::stencil[5][5]={{0,0,0,0,0},{0,0,0,0,0},{1,1,1,1,0},{1,1,1,1,0},{1,1,1,1,0}};
template<> const int IO<0,4>::stencil[5][5]={{0,0,0,0,0},{0,0,0,0,0},{1,1,1,0,0},{1,1,1,0,0},{1,1,1,0,0}};
template<> const int IO<1,0>::stencil[5][5]={{0,0,0,0,0},{0,0,1,1,1},{0,0,1,1,1},{0,0,1,1,1},{0,0,1,1,1}};
template<> const int IO<1,1>::stencil[5][5]={{0,0,0,0,0},{0,1,1,1,1},{0,1,1,1,1},{0,1,1,1,1},{0,1,1,1,1}};
template<> const int IO<1,2>::stencil[5][5]={{0,0,0,0,0},{1,1,1,1,1},{1,1,1,1,1},{1,1,1,1,1},{1,1,1,1,1}};
template<> const int IO<1,3>::stencil[5][5]={{0,0,0,0,0},{1,1,1,1,0},{1,1,1,1,0},{1,1,1,1,0},{1,1,1,1,0}};
template<> const int IO<1,4>::stencil[5][5]={{0,0,0,0,0},{1,1,1,0,0},{1,1,1,0,0},{1,1,1,0,0},{1,1,1,0,0}};
template<> const int IO<2,0>::stencil[5][5]={{0,0,1,1,1},{0,0,1,1,1},{0,0,1,1,1},{0,0,1,1,1},{0,0,1,1,1}};
template<> const int IO<2,1>::stencil[5][5]={{0,1,1,1,1},{0,1,1,1,1},{0,1,1,1,1},{0,1,1,1,1},{0,1,1,1,1}};
template<> const int IO<2,2>::stencil[5][5]={{1,1,1,1,1},{1,1,1,1,1},{1,1,1,1,1},{1,1,1,1,1},{1,1,1,1,1}};
template<> const int IO<2,3>::stencil[5][5]={{1,1,1,1,0},{1,1,1,1,0},{1,1,1,1,0},{1,1,1,1,0},{1,1,1,1,0}};
template<> const int IO<2,4>::stencil[5][5]={{1,1,1,0,0},{1,1,1,0,0},{1,1,1,0,0},{1,1,1,0,0},{1,1,1,0,0}};
template<> const int IO<3,0>::stencil[5][5]={{0,0,1,1,1},{0,0,1,1,1},{0,0,1,1,1},{0,0,1,1,1},{0,0,0,0,0}};
template<> const int IO<3,1>::stencil[5][5]={{0,1,1,1,1},{0,1,1,1,1},{0,1,1,1,1},{0,1,1,1,1},{0,0,0,0,0}};
template<> const int IO<3,2>::stencil[5][5]={{1,1,1,1,1},{1,1,1,1,1},{1,1,1,1,1},{1,1,1,1,1},{0,0,0,0,0}};
template<> const int IO<3,3>::stencil[5][5]={{1,1,1,1,0},{1,1,1,1,0},{1,1,1,1,0},{1,1,1,1,0},{0,0,0,0,0}};
template<> const int IO<3,4>::stencil[5][5]={{1,1,1,0,0},{1,1,1,0,0},{1,1,1,0,0},{1,1,1,0,0},{0,0,0,0,0}};
template<> const int IO<4,0>::stencil[5][5]={{0,0,1,1,1},{0,0,1,1,1},{0,0,1,1,1},{0,0,0,0,0},{0,0,0,0,0}};
template<> const int IO<4,1>::stencil[5][5]={{0,1,1,1,1},{0,1,1,1,1},{0,1,1,1,1},{0,0,0,0,0},{0,0,0,0,0}};
template<> const int IO<4,2>::stencil[5][5]={{1,1,1,1,1},{1,1,1,1,1},{1,1,1,1,1},{0,0,0,0,0},{0,0,0,0,0}};
template<> const int IO<4,3>::stencil[5][5]={{1,1,1,1,0},{1,1,1,1,0},{1,1,1,1,0},{0,0,0,0,0},{0,0,0,0,0}};
template<> const int IO<4,4>::stencil[5][5]={{1,1,1,0,0},{1,1,1,0,0},{1,1,1,0,0},{0,0,0,0,0},{0,0,0,0,0}};


template<> const int IO<0,0>::counter=9;
template<> const int IO<0,1>::counter=12;
template<> const int IO<0,2>::counter=15;
template<> const int IO<0,3>::counter=12;
template<> const int IO<0,4>::counter=9;
template<> const int IO<1,0>::counter=12;
template<> const int IO<1,1>::counter=16;
template<> const int IO<1,2>::counter=20;
template<> const int IO<1,3>::counter=16;
template<> const int IO<1,4>::counter=12;
template<> const int IO<2,0>::counter=15;
template<> const int IO<2,1>::counter=20;
template<> const int IO<2,2>::counter=25;
template<> const int IO<2,3>::counter=20;
template<> const int IO<2,4>::counter=15;
template<> const int IO<3,0>::counter=12;
template<> const int IO<3,1>::counter=16;
template<> const int IO<3,2>::counter=20;
template<> const int IO<3,3>::counter=16;
template<> const int IO<3,4>::counter=12;
template<> const int IO<4,0>::counter=9;
template<> const int IO<4,1>::counter=12;
template<> const int IO<4,2>::counter=15;
template<> const int IO<4,3>::counter=12;
template<> const int IO<4,4>::counter=9;



template<int ytype, int xtype>
class TPS
{
public:
	static const float stencil[5][5];

	template <class CNEI>
	float get(int index,CNEI &nei,float* src_data)
	{
		float sum=0.0f;
		if(stencil[0][0]!=0.0f)	sum+=src_data[nei.ll(nei.tt(index))]*stencil[0][0];
		if(stencil[0][1]!=0.0f)	sum+=src_data[nei.l(nei.tt(index))]*stencil[0][1];
		if(stencil[0][2]!=0.0f)	sum+=src_data[nei.tt(index)]*stencil[0][2];
		if(stencil[0][3]!=0.0f)	sum+=src_data[nei.r(nei.tt(index))]*stencil[0][3];
		if(stencil[0][4]!=0.0f)	sum+=src_data[nei.rr(nei.tt(index))]*stencil[0][4];
		if(stencil[1][0]!=0.0f)	sum+=src_data[nei.ll(nei.t(index))]*stencil[1][0];
		if(stencil[1][1]!=0.0f)	sum+=src_data[nei.l(nei.t(index))]*stencil[1][1];
		if(stencil[1][2]!=0.0f)	sum+=src_data[nei.t(index)]*stencil[1][2];
		if(stencil[1][3]!=0.0f)	sum+=src_data[nei.r(nei.t(index))]*stencil[1][3];
		if(stencil[1][4]!=0.0f)	sum+=src_data[nei.rr(nei.t(index))]*stencil[1][4];
		if(stencil[2][0]!=0.0f)	sum+=src_data[nei.ll(index)]*stencil[2][0];
		if(stencil[2][1]!=0.0f)	sum+=src_data[nei.l(index)]*stencil[2][1];
		if(stencil[2][2]!=0.0f)	sum+=src_data[index]*stencil[2][2];
		if(stencil[2][3]!=0.0f)	sum+=src_data[nei.r(index)]*stencil[2][3];
		if(stencil[2][4]!=0.0f)	sum+=src_data[nei.rr(index)]*stencil[2][4];
		if(stencil[3][0]!=0.0f)	sum+=src_data[nei.ll(nei.b(index))]*stencil[3][0];
		if(stencil[3][1]!=0.0f)	sum+=src_data[nei.l(nei.b(index))]*stencil[3][1];
		if(stencil[3][2]!=0.0f)	sum+=src_data[nei.b(index)]*stencil[3][2];
		if(stencil[3][3]!=0.0f)	sum+=src_data[nei.r(nei.b(index))]*stencil[3][3];
		if(stencil[3][4]!=0.0f)	sum+=src_data[nei.rr(nei.b(index))]*stencil[3][4];
		if(stencil[4][0]!=0.0f)	sum+=src_data[nei.ll(nei.bb(index))]*stencil[4][0];
		if(stencil[4][1]!=0.0f)	sum+=src_data[nei.l(nei.bb(index))]*stencil[4][1];
		if(stencil[4][2]!=0.0f)	sum+=src_data[nei.bb(index)]*stencil[4][2];
		if(stencil[4][3]!=0.0f)	sum+=src_data[nei.r(nei.bb(index))]*stencil[4][3];
		if(stencil[4][4]!=0.0f)	sum+=src_data[nei.rr(nei.bb(index))]*stencil[4][4];

		return sum;
	}

	template <class CNEI>
	void update(int index,CNEI &nei,float d, float* dst_data)
	{
		if(stencil[0][0]!=0.0f)	dst_data[nei.ll(nei.tt(index))]+=stencil[0][0]*d;
		if(stencil[0][1]!=0.0f)	dst_data[nei.l(nei.tt(index))]+=stencil[0][1]*d;
		if(stencil[0][2]!=0.0f)	dst_data[nei.tt(index)]+=stencil[0][2]*d;
		if(stencil[0][3]!=0.0f)	dst_data[nei.r(nei.tt(index))]+=stencil[0][3]*d;
		if(stencil[0][4]!=0.0f)	dst_data[nei.rr(nei.tt(index))]+=stencil[0][4]*d;
		if(stencil[1][0]!=0.0f)	dst_data[nei.ll(nei.t(index))]+=stencil[1][0]*d;
		if(stencil[1][1]!=0.0f)	dst_data[nei.l(nei.t(index))]+=stencil[1][1]*d;
		if(stencil[1][2]!=0.0f)	dst_data[nei.t(index)]+=stencil[1][2]*d;
		if(stencil[1][3]!=0.0f)	dst_data[nei.r(nei.t(index))]+=stencil[1][3]*d;
		if(stencil[1][4]!=0.0f)	dst_data[nei.rr(nei.t(index))]+=stencil[1][4]*d;
		if(stencil[2][0]!=0.0f)	dst_data[nei.ll(index)]+=stencil[2][0]*d;
		if(stencil[2][1]!=0.0f)	dst_data[nei.l(index)]+=stencil[2][1]*d;
		if(stencil[2][2]!=0.0f)	dst_data[index]+=stencil[2][2]*d;
		if(stencil[2][3]!=0.0f)	dst_data[nei.r(index)]+=stencil[2][3]*d;
		if(stencil[2][4]!=0.0f)	dst_data[nei.rr(index)]+=stencil[2][4]*d;
		if(stencil[3][0]!=0.0f)	dst_data[nei.ll(nei.b(index))]+=stencil[3][0]*d;
		if(stencil[3][1]!=0.0f)	dst_data[nei.l(nei.b(index))]+=stencil[3][1]*d;
		if(stencil[3][2]!=0.0f)	dst_data[nei.b(index)]+=stencil[3][2]*d;
		if(stencil[3][3]!=0.0f)	dst_data[nei.r(nei.b(index))]+=stencil[3][3]*d;
		if(stencil[3][4]!=0.0f)	dst_data[nei.rr(nei.b(index))]+=stencil[3][4]*d;
		if(stencil[4][0]!=0.0f)	dst_data[nei.ll(nei.bb(index))]+=stencil[4][0]*d;
		if(stencil[4][1]!=0.0f)	dst_data[nei.l(nei.bb(index))]+=stencil[4][1]*d;
		if(stencil[4][2]!=0.0f)	dst_data[nei.bb(index)]+=stencil[4][2]*d;
		if(stencil[4][3]!=0.0f)	dst_data[nei.r(nei.bb(index))]+=stencil[4][3]*d;
		if(stencil[4][4]!=0.0f)	dst_data[nei.rr(nei.bb(index))]+=stencil[4][4]*d;
	}
};

template<> const float TPS<0,0>::stencil[5][5]={{0,0,0,0,0},{0,0,0,0,0},{0,0,8,-8,2},{0,0,-8,4,0},{0,0,2,0,0}};
template<> const float TPS<0,1>::stencil[5][5]={{0,0,0,0,0},{0,0,0,0,0},{0,-8,20,-12,2},{0,4,-12,4,0},{0,0,2,0,0}};
template<> const float TPS<0,2>::stencil[5][5]={{0,0,0,0,0},{0,0,0,0,0},{2,-12,22,-12,2},{0,4,-12,4,0},{0,0,2,0,0}};
template<> const float TPS<0,3>::stencil[5][5]={{0,0,0,0,0},{0,0,0,0,0},{2,-12,20,-8,0},{0,4,-12,4,0},{0,0,2,0,0}};
template<> const float TPS<0,4>::stencil[5][5]={{0,0,0,0,0},{0,0,0,0,0},{2,-8,8,0,0},{0,4,-8,0,0},{0,0,2,0,0}};
template<> const float TPS<1,0>::stencil[5][5]={{0,0,0,0,0},{0,0,-8,4,0},{0,0,20,-12,2},{0,0,-12,4,0},{0,0,2,0,0}};
template<> const float TPS<1,1>::stencil[5][5]={{0,0,0,0,0},{0,4,-12,4,0},{0,-12,36,-16,2},{0,4,-16,4,0},{0,0,2,0,0}};
template<> const float TPS<1,2>::stencil[5][5]={{0,0,0,0,0},{0,4,-12,4,0},{2,-16,38,-16,2},{0,4,-16,4,0},{0,0,2,0,0}};
template<> const float TPS<1,3>::stencil[5][5]={{0,0,0,0,0},{0,4,-12,4,0},{2,-16,36,-12,0},{0,4,-16,4,0},{0,0,2,0,0}};
template<> const float TPS<1,4>::stencil[5][5]={{0,0,0,0,0},{0,4,-8,0,0},{2,-12,20,0,0},{0,4,-12,0,0},{0,0,2,0,0}};
template<> const float TPS<2,0>::stencil[5][5]={{0,0,2,0,0},{0,0,-12,4,0},{0,0,22,-12,2},{0,0,-12,4,0},{0,0,2,0,0}};
template<> const float TPS<2,1>::stencil[5][5]={{0,0,2,0,0},{0,4,-16,4,0},{0,-12,38,-16,2},{0,4,-16,4,0},{0,0,2,0,0}};
template<> const float TPS<2,2>::stencil[5][5]={{0,0,2,0,0},{0,4,-16,4,0},{2,-16,40,-16,2},{0,4,-16,4,0},{0,0,2,0,0}};
template<> const float TPS<2,3>::stencil[5][5]={{0,0,2,0,0},{0,4,-16,4,0},{2,-16,38,-12,0},{0,4,-16,4,0},{0,0,2,0,0}};
template<> const float TPS<2,4>::stencil[5][5]={{0,0,2,0,0},{0,4,-12,0,0},{2,-12,22,0,0},{0,4,-12,0,0},{0,0,2,0,0}};
template<> const float TPS<3,0>::stencil[5][5]={{0,0,2,0,0},{0,0,-12,4,0},{0,0,20,-12,2},{0,0,-8,4,0},{0,0,0,0,0}};
template<> const float TPS<3,1>::stencil[5][5]={{0,0,2,0,0},{0,4,-16,4,0},{0,-12,36,-16,2},{0,4,-12,4,0},{0,0,0,0,0}};
template<> const float TPS<3,2>::stencil[5][5]={{0,0,2,0,0},{0,4,-16,4,0},{2,-16,38,-16,2},{0,4,-12,4,0},{0,0,0,0,0}};
template<> const float TPS<3,3>::stencil[5][5]={{0,0,2,0,0},{0,4,-16,4,0},{2,-16,36,-12,0},{0,4,-12,4,0},{0,0,0,0,0}};
template<> const float TPS<3,4>::stencil[5][5]={{0,0,2,0,0},{0,4,-12,0,0},{2,-12,20,0,0},{0,4,-8,0,0},{0,0,0,0,0}};
template<> const float TPS<4,0>::stencil[5][5]={{0,0,2,0,0},{0,0,-8,4,0},{0,0,8,-8,2},{0,0,0,0,0},{0,0,0,0,0}};
template<> const float TPS<4,1>::stencil[5][5]={{0,0,2,0,0},{0,4,-12,4,0},{0,-8,20,-12,2},{0,0,0,0,0},{0,0,0,0,0}};
template<> const float TPS<4,2>::stencil[5][5]={{0,0,2,0,0},{0,4,-12,4,0},{2,-12,22,-12,2},{0,0,0,0,0},{0,0,0,0,0}};
template<> const float TPS<4,3>::stencil[5][5]={{0,0,2,0,0},{0,4,-12,4,0},{2,-12,20,-8,0},{0,0,0,0,0},{0,0,0,0,0}};
template<> const float TPS<4,4>::stencil[5][5]={{0,0,2,0,0},{0,4,-8,0,0},{2,-8,8,0,0},{0,0,0,0,0},{0,0,0,0,0}};



template<int ytype, int xtype>
class Oper
{
public:
	IO<ytype, xtype> io;
	TPS<ytype, xtype> tps;
};
