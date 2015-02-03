#include "QuadraticPath.h"

CQuadraticPath::CQuadraticPath(int layer_index,cv::Mat& vector,cv::Mat& qpath,bool gpu_flag):_layer_index(layer_index),_vector(vector),_qpath(qpath),_gpu_flag(gpu_flag)
{
	w=vector.cols;
	h=vector.rows;
	_levels=log((float)MIN(_vector.cols,_vector.rows))/log(2.0f)-log(32.0f)/log(2.0f)+1;
}

void CQuadraticPath::run()
{
	 clock_t start, finish;
	 start=clock();
	/* if(_gpu_flag)
		 run_cuda();
	 else*/
	     run_cpu();
	finish=clock();
	_runtime= (float)(finish - start)* 1000/ CLOCKS_PER_SEC; 
	emit sigFinished(_layer_index);
}


void CQuadraticPath::run_cpu()
{
	//prepare
	int size=w*h;
	float *j_opt=new float[size*4];
	float *p_opt=new float[size];
	float *vx=new float[size];
	float *vy=new float[size];

#pragma omp parallel for
	for(int y=0;y<h;y++)
		for (int x=0;x<w;x++)
		{
			vx[y*w+x]=_vector.at<Vec3f>(y,x)[0];
			vy[y*w+x]=_vector.at<Vec3f>(y,x)[1];

			float j0[4],j1[4];			
			float vx_x,vy_x;
			float vx_y,vy_y;
			if(x==0)
			{
				vx_x=_vector.at<Vec3f>(y,x+1)[0]-_vector.at<Vec3f>(y,x)[0];
				vy_x=_vector.at<Vec3f>(y,x+1)[1]-_vector.at<Vec3f>(y,x)[1];
			}
			else 
			{
				vx_x=_vector.at<Vec3f>(y,x)[0]-_vector.at<cv::Vec3f>(y,x-1)[0];
				vy_x=_vector.at<Vec3f>(y,x)[1]-_vector.at<cv::Vec3f>(y,x-1)[1];
			}

			j0[0]=1.0f-vx_x;
			j0[2]=-vy_x;
			j1[0]=1.0f+vx_x;
			j1[2]=vy_x;

			if(y==0)
			{
				vx_y=_vector.at<Vec3f>(y+1,x)[0]-_vector.at<Vec3f>(y,x)[0];
				vy_y=_vector.at<Vec3f>(y+1,x)[1]-_vector.at<Vec3f>(y,x)[1];

			}
			else 
			{
				vx_y=_vector.at<Vec3f>(y,x)[0]-_vector.at<Vec3f>(y-1,x)[0];
				vy_y=_vector.at<Vec3f>(y,x)[1]-_vector.at<Vec3f>(y-1,x)[1];
			}


			j0[1]=-vx_y;
			j0[3]=1.0f-vy_y;
			j1[1]=vx_y;
			j1[3]=1.0f+vy_y;

			//optimal J
			float nj0[4],nj1[4];
			float la0,lb0,la1,lb1;
			la0=sqrt(j0[0]*j0[0]+j0[2]*j0[2]);	
			lb0=sqrt(j0[1]*j0[1]+j0[3]*j0[3]);
			nj0[0]=j0[0]/la0;
			nj0[2]=j0[2]/la0;
			nj0[1]=j0[1]/lb0;
			nj0[3]=j0[3]/lb0;

			la1=sqrt(j1[0]*j1[0]+j1[2]*j1[2]);	
			lb1=sqrt(j1[1]*j1[1]+j1[3]*j1[3]);
			nj1[0]=j1[0]/la1;
			nj1[2]=j1[2]/la1;
			nj1[1]=j1[1]/lb1;
			nj1[3]=j1[3]/lb1;

			//rotate
			float nj_opt[4];
			for(int i=0;i<4;i++)
				nj_opt[i]=nj0[i]+nj1[i];
			float la_opt=sqrt(nj_opt[0]*nj_opt[0]+nj_opt[2]*nj_opt[2]);
			float lb_opt=sqrt(nj_opt[1]*nj_opt[1]+nj_opt[3]*nj_opt[3]);			


			nj_opt[0]/=la_opt;
			nj_opt[2]/=la_opt;
			nj_opt[1]/=lb_opt;
			nj_opt[3]/=lb_opt;

			//scale
			la_opt=sqrt(la0*la1);
			lb_opt=sqrt(lb0*lb1);

			int index=y*w*4+x*4;
			j_opt[index+0]=nj_opt[0]*la_opt;
			j_opt[index+2]=nj_opt[2]*la_opt;
			j_opt[index+1]=nj_opt[1]*lb_opt;
			j_opt[index+3]=nj_opt[3]*lb_opt;

			//p
			float dx=_vector.at<Vec3f>(y,x)[0];
			float dy=_vector.at<Vec3f>(y,x)[1];
			float dis=sqrt(dx*dx+dy*dy);
			if(dis<1)
				p_opt[y*w+x]=1.0f-dis;
			else
				p_opt[y*w+x]=0.0f;	

					
		}
	

		//matrix	
		float *A=new float[5*size];
		_INTEGER_t  *columns=new _INTEGER_t [5*size];
		_INTEGER_t  *rowindex=new _INTEGER_t [size+1];
		float *Bx=new float[size];
		float *By=new float[size];		
		float *X=new float[size];
		float *Y=new float[size];

		_INTEGER_t  nNonZeros=0;

		memset(A,0,5*size*sizeof(float));
		memset(columns,0,5*size*sizeof(int));
		memset(rowindex,0,(size+1)*sizeof(int));
		memset(Bx,0,size*sizeof(float));
		memset(By,0,size*sizeof(float));		
		memset(X,0,size*sizeof(float));
		memset(Y,0,size*sizeof(float));

		rowindex[0]=0;

		for(int y=0;y<h;y++)
			for (int x=0;x<w;x++)
			{
				float a[5];
				a[0]=a[1]=a[2]=a[3]=a[4]=0.0f;

				int ii=y*w+x;
				if(y-1>=0) 
				{
					a[2]+=1.0f;
					a[0]-=1.0f;	
					Bx[ii]+=j_opt[y*w*4+x*4+1];
					By[ii]+=j_opt[y*w*4+x*4+3]-1.0f;		
				}
				if(x-1>=0) 
				{
					a[2]+=1.0f;
					a[1]-=1.0f;	
					Bx[ii]+=j_opt[y*w*4+x*4+0]-1.0f;
					By[ii]+=j_opt[y*w*4+x*4+2];

				}
				if(x+1<w) 
				{
					a[2]+=1.0f;
					a[3]-=1.0f;	
					Bx[ii]-=j_opt[y*w*4+(x+1)*4+0]-1.0f;
					By[ii]-=j_opt[y*w*4+(x+1)*4+2];
				}		
				if(y+1<h) 
				{
					a[2]+=1.0f;
					a[4]-=1.0f;	
					Bx[ii]-=j_opt[(y+1)*w*4+x*4+1];
					By[ii]-=j_opt[(y+1)*w*4+x*4+3]-1.0f;		
				}

				//p
				a[2]+=p_opt[y*w+x];
				//put into A
				if(a[0]!=0)
				{
					A[nNonZeros]=a[0];
					columns[nNonZeros]=ii-w;
					nNonZeros++;
				}
				if(a[1]!=0)
				{
					A[nNonZeros]=a[1];
					columns[nNonZeros]=ii-1;
					nNonZeros++;
				}
				if(a[2]!=0)
				{
					A[nNonZeros]=a[2];
					columns[nNonZeros]=ii;
					nNonZeros++;
				}
				if(a[3]!=0)
				{
					A[nNonZeros]=a[3];
					columns[nNonZeros]=ii+1;
					nNonZeros++;
				}
				if(a[4]!=0)
				{
					A[nNonZeros]=a[4];
					columns[nNonZeros]=ii+w;
					nNonZeros++;
				}
				rowindex[ii+1]=nNonZeros;
			}

			//SOLVER
			_INTEGER_t error;
			_MKL_DSS_HANDLE_t solver;
			_INTEGER_t opt=MKL_DSS_MSG_LVL_WARNING + MKL_DSS_TERM_LVL_ERROR + MKL_DSS_SINGLE_PRECISION + MKL_DSS_ZERO_BASED_INDEXING;
			_INTEGER_t sym=MKL_DSS_NON_SYMMETRIC;
			_INTEGER_t typ=MKL_DSS_POSITIVE_DEFINITE;
			_INTEGER_t ord=MKL_DSS_AUTO_ORDER;
			_INTEGER_t sov=MKL_DSS_DEFAULTS;
			_INTEGER_t nRhs = 1;
			_INTEGER_t size_l=size;

			error = dss_create(solver, opt);
			error = dss_define_structure(solver,sym, rowindex, size_l, size_l,columns, nNonZeros);
			error = dss_reorder( solver,ord, 0);
			error = dss_factor_real( solver, typ, A );
			error = dss_solve_real( solver, sov, Bx, nRhs, X );
			error = dss_solve_real( solver, sov, By, nRhs, Y );		
			error = dss_delete( solver, opt );

			//paste	
			_qpath=cv::Mat(_vector.rows,_vector.cols,CV_32FC3);
			#pragma omp parallel for
			for(int y=0;y<h;y++)
				for(int x=0;x<w;x++)				
					_qpath.at<Vec3f>(y,x)=Vec3f(X[y*w+x],Y[y*w+x],0);		

			delete[] j_opt;		
			delete[] p_opt;
			delete[] A;
			delete[] Bx;
			delete[] By;
			delete[] X;
			delete[] Y;
			delete[] columns;
			delete[] rowindex;
			delete[] vx;
			delete[] vy;

			
};

void CQuadraticPath::run_cuda()
{
	cv::Mat* vector=new cv::Mat[_levels];
	cv::Mat* qpath=new cv::Mat[_levels];
	
	vector[0]=_vector.clone();

	for (int l=_levels-1;l>=0;l--)
	{
		float fa=1.0f/pow(2.0f,l);
		cv::resize(vector[0],vector[l],cvSize(0,0),fa,fa,INTER_LINEAR);
		vector[l]=vector[l]*fa;
		
		if(l<_levels-1)	
		{
			cv::resize(qpath[l+1],qpath[l],cvSize(vector[l].cols,vector[l].rows),0,0, INTER_LINEAR);
			qpath[l]=qpath[l]*2.0f;
		}
		else
			qpath[l]=Mat::zeros(vector[l].rows,vector[l].cols,CV_32FC3);
	
		run_level(vector[l],qpath[l]);	
	}
	_qpath=qpath[0].clone();
}

void CQuadraticPath::run_level(cv::Mat& _vector, cv::Mat& _qpath)
{
	//prepare
	w=_vector.cols;
	h=_vector.rows;
	int size=w*h;
	float *j_opt=new float[size*4];
	float *p_opt=new float[size];

#pragma omp parallel for
	for(int y=0;y<h;y++)
		for (int x=0;x<w;x++)
		{
			float j0[4],j1[4];			
			float vx_x,vy_x;
			float vx_y,vy_y;
			if(x==0)
			{
				vx_x=_vector.at<Vec3f>(y,x+1)[0]-_vector.at<Vec3f>(y,x)[0];
				vy_x=_vector.at<Vec3f>(y,x+1)[1]-_vector.at<Vec3f>(y,x)[1];
			}
			else 
			{
				vx_x=_vector.at<Vec3f>(y,x)[0]-_vector.at<cv::Vec3f>(y,x-1)[0];
				vy_x=_vector.at<Vec3f>(y,x)[1]-_vector.at<cv::Vec3f>(y,x-1)[1];
			}
			
			j0[0]=1.0f-vx_x;
			j0[2]=-vy_x;
			j1[0]=1.0f+vx_x;
			j1[2]=vy_x;

			if(y==0)
			{
				vx_y=_vector.at<Vec3f>(y+1,x)[0]-_vector.at<Vec3f>(y,x)[0];
				vy_y=_vector.at<Vec3f>(y+1,x)[1]-_vector.at<Vec3f>(y,x)[1];

			}
			else 
			{
				vx_y=_vector.at<Vec3f>(y,x)[0]-_vector.at<Vec3f>(y-1,x)[0];
				vy_y=_vector.at<Vec3f>(y,x)[1]-_vector.at<Vec3f>(y-1,x)[1];
			}
			

			j0[1]=-vx_y;
			j0[3]=1.0f-vy_y;
			j1[1]=vx_y;
			j1[3]=1.0f+vy_y;

			//optimal J
			float nj0[4],nj1[4];
			float la0,lb0,la1,lb1;
			la0=sqrt(j0[0]*j0[0]+j0[2]*j0[2]);	
			lb0=sqrt(j0[1]*j0[1]+j0[3]*j0[3]);
			nj0[0]=j0[0]/la0;
			nj0[2]=j0[2]/la0;
			nj0[1]=j0[1]/lb0;
			nj0[3]=j0[3]/lb0;

			la1=sqrt(j1[0]*j1[0]+j1[2]*j1[2]);	
			lb1=sqrt(j1[1]*j1[1]+j1[3]*j1[3]);
			nj1[0]=j1[0]/la1;
			nj1[2]=j1[2]/la1;
			nj1[1]=j1[1]/lb1;
			nj1[3]=j1[3]/lb1;

			//rotate
			float nj_opt[4];
			for(int i=0;i<4;i++)
				nj_opt[i]=nj0[i]+nj1[i];
			float la_opt=sqrt(nj_opt[0]*nj_opt[0]+nj_opt[2]*nj_opt[2]);
			float lb_opt=sqrt(nj_opt[1]*nj_opt[1]+nj_opt[3]*nj_opt[3]);			


			nj_opt[0]/=la_opt;
			nj_opt[2]/=la_opt;
			nj_opt[1]/=lb_opt;
			nj_opt[3]/=lb_opt;

			//scale
			la_opt=sqrt(la0*la1);
			lb_opt=sqrt(lb0*lb1);

			int index=y*w*4+x*4;
			j_opt[index+0]=nj_opt[0]*la_opt;
			j_opt[index+2]=nj_opt[2]*la_opt;
			j_opt[index+1]=nj_opt[1]*lb_opt;
			j_opt[index+3]=nj_opt[3]*lb_opt;

			//p
			float vx=_vector.at<Vec3f>(y,x)[0];
			float vy=_vector.at<Vec3f>(y,x)[1];
			float dis=sqrt(vx*vx+vy*vy);
			if(dis<1)
				p_opt[y*w+x]=1.0f-dis;
			else
				p_opt[y*w+x]=0.0f;			
		}


		//matrix	
		cusp::csr_matrix<int,float,cusp::host_memory> A(size,size,5*size-w*2-h*2);
		cusp::array1d<float, cusp::host_memory> X(A.num_rows, 0);
		cusp::array1d<float, cusp::host_memory> Bx(A.num_rows, 0);
		cusp::array1d<float, cusp::host_memory> Y(A.num_rows, 0);
		cusp::array1d<float, cusp::host_memory> By(A.num_rows, 0);

		int nNonZeros=0;
		A.row_offsets[0]=0;

		for(int y=0;y<h;y++)
			for (int x=0;x<w;x++)
			{
				float a[5];
				a[0]=a[1]=a[2]=a[3]=a[4]=0.0f;

				int ii=y*w+x;
				X[ii]=_qpath.at<Vec3f>(y,x)[0];
				Y[ii]=_qpath.at<Vec3f>(y,x)[1];
				if(y-1>=0) 
				{
					a[2]+=1.0f;
					a[0]-=1.0f;	
					Bx[ii]+=j_opt[y*w*4+x*4+1];
					By[ii]+=j_opt[y*w*4+x*4+3]-1.0f;		
				}
				if(x-1>=0) 
				{
					a[2]+=1.0f;
					a[1]-=1.0f;	
					Bx[ii]+=j_opt[y*w*4+x*4+0]-1.0f;
					By[ii]+=j_opt[y*w*4+x*4+2];

				}
				if(x+1<w) 
				{
					a[2]+=1.0f;
					a[3]-=1.0f;	
					Bx[ii]-=j_opt[y*w*4+(x+1)*4+0]-1.0f;
					By[ii]-=j_opt[y*w*4+(x+1)*4+2];
				}		
				if(y+1<h) 
				{
					a[2]+=1.0f;
					a[4]-=1.0f;	
					Bx[ii]-=j_opt[(y+1)*w*4+x*4+1];
					By[ii]-=j_opt[(y+1)*w*4+x*4+3]-1.0f;		
				}

				//p
				a[2]+=p_opt[y*w+x];
				//put into A
				if(a[0]!=0)
				{
					A.values[nNonZeros]=a[0];
					A.column_indices[nNonZeros]=ii-w;
					nNonZeros++;
				}
				if(a[1]!=0)
				{
					A.values[nNonZeros]=a[1];
					A.column_indices[nNonZeros]=ii-1;
					nNonZeros++;
				}
				if(a[2]!=0)
				{
					A.values[nNonZeros]=a[2];
					A.column_indices[nNonZeros]=ii;
					nNonZeros++;
				}
				if(a[3]!=0)
				{
					A.values[nNonZeros]=a[3];
					A.column_indices[nNonZeros]=ii+1;
					nNonZeros++;
				}
				if(a[4]!=0)
				{
					A.values[nNonZeros]=a[4];
					A.column_indices[nNonZeros]=ii+w;
					nNonZeros++;
				}
				A.row_offsets[ii+1]=nNonZeros;

			}
		
		
			solve(A,Bx,X);
			solve(A,By,Y);		
		
			//paste		
			#pragma omp parallel for
			for(int y=0;y<h;y++)
				for(int x=0;x<w;x++)				
					_qpath.at<Vec3f>(y,x)=Vec3f(X[y*w+x],Y[y*w+x],0);				
			
			delete[] j_opt;		
			delete[] p_opt;

}


CQuadraticPath::~CQuadraticPath(void)
{
	
}

