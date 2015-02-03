#include "PoissonExt.h"


CPoissonExt::CPoissonExt(int layer_index, cv::Mat& vector, cv::Mat& extends1, cv::Mat& extends2,bool gpu_flag):_layer_index(layer_index),_vector(vector),_extends1(extends1),_extends2(extends2),_gpu_flag(gpu_flag)
{
	w=vector.cols;
	h=vector.rows;
	ex=(_extends1.cols-w)/2;

	Rect sourceRect(ex,ex,w, h);

	_image1=_extends1(sourceRect).clone();
	_image2=_extends2(sourceRect).clone();

	type=new int[(w+2*ex)*(h+ex*2)];
	index=new int[(w+2*ex)*(h+ex*2)];
}

// void CPoissonExt::Result_mesh()
// {
// 
// 	QFile file("mesh.m");
// 
// 	if (file.open(QFile::WriteOnly | QFile::Truncate))
// 	{
// 		QTextStream out(&file);	
// 		QString line;
// 
// 		for(int y=0;y<h;y++)
// 			for(int x=0;x<w;x++)
// 			{
// 				Vec3f v;
// 				v=_vector.at<Vec3f>(y,x);
// 				Vec4f rgba1=BilineaGetColor_clamp<Vec4b,Vec4f>(_image1,x-v[0],y-v[1]);
// 				Vec4f rgba2=BilineaGetColor_clamp<Vec4b,Vec4f>(_image2,x+v[0],y+v[1]);
// 
// 				line.sprintf("Vertex %d %f %f 0 {Opos=(%f %f 0) rgb=(%f %f %f) Orgb=(%f %f %f)}\n",y*w+x+1,(x+v[0])/(float)(w-1),1.0-(y+v[1])/(float)(h-1),(x-v[0])/(float)(w-1),1.0-(y-v[1])/(float)(h-1),
// 					rgba2[2]/255.0f,rgba2[1]/255.0f,rgba2[0]/255.0f,rgba1[2]/255.0f,rgba1[1]/255.0f,rgba1[0]/255.0f);
// 				out<<line;
// 			}	
// 
// 			for (int y=0; y<h-1; y++)
// 				for (int x=0; x<w-1; x++)
// 				{
// 					line.sprintf("Face %d  %d %d %d %d\n", y*(w-1)+x+1, y*w+x+1, (y+1)*w+x+1, (y+1)*w+x+1+1, y*w+x+1+1);
// 					out<<line;
// 				}
// 
// 	}
// 	file.flush(); 
// 	file.close(); 
// 
// }
// 



void CPoissonExt::run()
{
	clock_t start, finish;
	start=clock();

	int size;
	size=prepare(1,_extends1);
	/*if(_gpu_flag)
		poissonExtend_cuda(_extends1,size);
	else*/
		poissonExtend(_extends1,size);
	size=prepare(2,_extends2);
   /*	if(_gpu_flag)
   		poissonExtend_cuda(_extends2,size);
   	else*/
		poissonExtend(_extends2,size);

	finish=clock();
	_runtime= (float)(finish - start) * 1000 / CLOCKS_PER_SEC;
	emit sigFinished(_layer_index);
}

CPoissonExt::~CPoissonExt(void)
{
	delete[] type;
	delete[] index;
}

int CPoissonExt:: prepare(int side, cv::Mat &extends)
{
	int size=0;
	int sign;
	cv::Mat *image;
	if(side==1)
		sign=1,image=&_image2;
	else
		sign=-1,image=&_image1;


	for(int y=0;y<h+ex*2;y++)
	 for (int x=0;x<w+ex*2;x++)
	{
		int ii=y*(w+2*ex)+x;
	    if((extends.at<Vec4b>(y,x))[3]>0)
		{
			type[ii]=2;
			index[ii]=size++;
		}
		else
		{
			//4ÁÚÓò
			if(y>0&&(extends.at<Vec4b>(y-1,x))[3]>0)
			{
				type[ii]=1;
				index[ii]=size++;
				continue;
			}
			if(y<h+ex*2-1&&(extends.at<Vec4b>(y+1,x))[3]>0)
			{
				type[ii]=1;
				index[ii]=size++;
				continue;
			}
			if(x>0&&(extends.at<Vec4b>(y,x-1))[3]>0)
			{
				type[ii]=1;
				index[ii]=size++;
				continue;
			}
			if(x<w+ex*2-1&&(extends.at<Vec4b>(y,x+1))[3]>0)
			{
				type[ii]=1;
				index[ii]=size++;
				continue;
			}

			type[ii]=0;
		}
	}


	#pragma omp parallel for
	for(int y=0;y<h+ex*2;y++)
		for (int x=0;x<w+ex*2;x++)
		{

			int ii=y*(w+2*ex)+x;

			if(type[ii]==2)//outside
			{
				Vec3f q,p,v;

				q[0]=x-ex;
				q[1]=y-ex;
				p=q;
				v=BilineaGetColor_clamp<Vec3f,Vec3f>(_vector,p[0],p[1]);

				float alpha=0.8;

				for(int i=0;i<20;i++)
				{
					p=q+v*sign;
					v=alpha*BilineaGetColor_clamp<Vec3f,Vec3f>(_vector,p[0],p[1])+(1-alpha)*v;
				}

				q=p+v*sign;
				if(q[0]>=0&&q[1]>=0&&q[0]<w&&q[1]<h)
				{
					Vec4b rgba=BilineaGetColor_clamp<Vec4b,Vec4f>(*image,q[0],q[1]);
					if(rgba[3]==0)
						extends.at<Vec4b>(y,x)=rgba;
					else
						extends.at<Vec4b>(y,x)=Vec4b(255,0,255,0);
				}
				else
					extends.at<Vec4b>(y,x)=Vec4b(255,0,255,0);
			}
		}


	return size;
}

void CPoissonExt::poissonExtend(cv::Mat &dst,int size)
{

	cv::Mat gx=Mat::zeros(h+2*ex,w+2*ex,CV_32FC4);
	cv::Mat gy=Mat::zeros(h+2*ex,w+2*ex,CV_32FC4);

	#pragma omp parallel for
		for(int y=0;y<h+ex*2;y++)
			for (int x=0;x<w+ex*2;x++)
			{
				if(type[y*(w+2*ex)+x]>1)
				{
					Vec4f BGRA0;
					Vec4f BGRA1;
					BGRA1=dst.at<Vec4b>(y,x);

					if(x>0)
					{
						if(type[y*(w+2*ex)+x-1]>1)
						{
							BGRA0=dst.at<Vec4b>(y,x-1);

							if(BGRA0!=Vec4f(255,0,255,0)&&BGRA1!=Vec4f(255,0,255,0))
								gx.at<Vec4f>(y,x)=BGRA1-BGRA0;
						}
					}

					if(y>0)
					{
						if(type[(y-1)*(w+2*ex)+x]>1)
						{
							BGRA0=dst.at<Vec4b>(y-1,x);

							if(BGRA0!=Vec4f(255,0,255,0)&&BGRA1!=Vec4f(255,0,255,0))
								gy.at<Vec4f>(y,x)=BGRA1-BGRA0;
						}
					}
				}

			}
			//matrix
			float *A=new float[5*size];
			 _INTEGER_t *columns=new  _INTEGER_t[5*size];
			 _INTEGER_t *rowindex=new  _INTEGER_t[size+1];
			float *B0=new float[size];
			float *B1=new float[size];
			float *B2=new float[size];
			float *B3=new float[size];
			float *X0=new float[size];
			float *X1=new float[size];
			float *X2=new float[size];
			//float *X3=new float[size];
			_INTEGER_t  nNonZeros=0;

			memset(A,0,5*size*sizeof(float));
			memset(columns,0,5*size*sizeof(int));
			memset(rowindex,0,(size+1)*sizeof(int));
			memset(B0,0,size*sizeof(float));
			memset(B1,0,size*sizeof(float));
			memset(B2,0,size*sizeof(float));
			memset(B3,0,size*sizeof(float));
			memset(X0,0,size*sizeof(float));
			memset(X1,0,size*sizeof(float));
			memset(X2,0,size*sizeof(float));
			//memset(X3,0,size*sizeof(float));

			Vec4f BGRA;
			for(int y=0;y<h+ex*2;y++)
				for (int x=0;x<w+ex*2;x++)
				{
					float a[5];
					a[0]=a[1]=a[2]=a[3]=a[4]=0.0f;
					int ii=y*(w+2*ex)+x;

					switch(type[ii])
					{
					case 0://inside
						break;
					case 1://boundary
						a[2]+=1.0f;
						BGRA=dst.at<Vec4b>(y,x);
						B0[index[ii]]+=BGRA[0];
						B1[index[ii]]+=BGRA[1];
						B2[index[ii]]+=BGRA[2];
						//B3[index[ii]]+=BGRA[3];

					case 2://outside
						if(y-1>=0&&type[ii-(w+2*ex)]>0)
						{
							a[2]+=1.0f;
							a[0]-=1.0f;
							BGRA=gy.at<Vec4f>(y,x);
							B0[index[ii]]+=BGRA[0];
							B1[index[ii]]+=BGRA[1];
							B2[index[ii]]+=BGRA[2];
							//B3[index[ii]]+=BGRA[3];
						}
						if(x-1>=0&&type[ii-1]>0)
						{
							a[2]+=1.0f;
							a[1]-=1.0f;
							BGRA=gx.at<Vec4f>(y,x);
							B0[index[ii]]+=BGRA[0];
							B1[index[ii]]+=BGRA[1];
							B2[index[ii]]+=BGRA[2];
							//B3[index[ii]]+=BGRA[3];
						}
						if(x+1<w+2*ex&&type[ii+1]>0)
						{
							a[2]+=1.0f;
							a[3]-=1.0f;
							BGRA=gx.at<Vec4f>(y,x+1);
							B0[index[ii]]-=BGRA[0];
							B1[index[ii]]-=BGRA[1];
							B2[index[ii]]-=BGRA[2];
							//B3[index[ii]]-=BGRA[3];
						}
						if(y+1<h+2*ex&&type[ii+(w+2*ex)]>0)
						{
							a[2]+=1.0f;
							a[4]-=1.0f;
							BGRA=gy.at<Vec4f>(y+1,x);
							B0[index[ii]]-=BGRA[0];
							B1[index[ii]]-=BGRA[1];
							B2[index[ii]]-=BGRA[2];
							//B3[index[ii]]-=BGRA[3];
						}

						//put into A
						if(a[0]!=0)
						{
							A[nNonZeros]=a[0];
							columns[nNonZeros]=index[ii-(w+2*ex)];
							nNonZeros++;
						}
						if(a[1]!=0)
						{
							A[nNonZeros]=a[1];
							columns[nNonZeros]=index[ii-1];
							nNonZeros++;
						}
						if(a[2]!=0)
						{
							A[nNonZeros]=a[2];
							columns[nNonZeros]=index[ii];
							nNonZeros++;
						}
						if(a[3]!=0)
						{
							A[nNonZeros]=a[3];
							columns[nNonZeros]=index[ii+1];
							nNonZeros++;
						}
						if(a[4]!=0)
						{
							A[nNonZeros]=a[4];
							columns[nNonZeros]=index[ii+(w+2*ex)];
							nNonZeros++;
						}
						rowindex[index[ii]+1]=nNonZeros;

						break;
					}
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

				error= dss_create(solver, opt);
				error = dss_define_structure(solver,sym, rowindex, size_l, size_l,columns, nNonZeros);
				error = dss_reorder( solver,ord, 0);
				error = dss_factor_real( solver, typ, A );
				error = dss_solve_real( solver, sov, B0, nRhs, X0 );
				error = dss_solve_real( solver, sov, B1, nRhs, X1 );
				error = dss_solve_real( solver, sov, B2, nRhs, X2 );
				//error = dss_solve_real( solver, sov, B3, nRhs, X3 );
				error = dss_delete( solver, opt );


				//paste
				#pragma omp parallel for
				for(int y=0;y<h+ex*2;y++)
					for (int x=0;x<w+ex*2;x++)
					{
						int ii=y*(w+2*ex)+x;
						if(type[ii]>0)
						{
							int B=MIN(MAX(X0[index[ii]],0),255);
							int G=MIN(MAX(X1[index[ii]],0),255);
							int R=MIN(MAX(X2[index[ii]],0),255);
							int A=0;
							dst.at<Vec4b>(y,x)=Vec4b(B,G,R,A);

						}
					}


					delete[] A;
					delete[] B0;
					delete[] B1;
					delete[] B2;
					delete[] B3;
					delete[] X0;
					delete[] X1;
					delete[] X2;
					//delete[] X3;
					delete[] columns;
					delete[] rowindex;

}

void CPoissonExt::poissonExtend_cuda(cv::Mat &dst,int size)
{

	cv::Mat gx=Mat::zeros(h+2*ex,w+2*ex,CV_32FC4);
	cv::Mat gy=Mat::zeros(h+2*ex,w+2*ex,CV_32FC4);

	#pragma omp parallel for
		for(int y=0;y<h+ex*2;y++)
			for (int x=0;x<w+ex*2;x++)
			{
				if(type[y*(w+2*ex)+x]>1)
				{
					Vec4f BGRA0;
					Vec4f BGRA1;
					BGRA1=dst.at<Vec4b>(y,x);

					if(x>0)
					{
						if(type[y*(w+2*ex)+x-1]>1)
						{
							BGRA0=dst.at<Vec4b>(y,x-1);

							if(BGRA0!=Vec4f(255,0,255,0)&&BGRA1!=Vec4f(255,0,255,0))
								gx.at<Vec4f>(y,x)=BGRA1-BGRA0;

						}
					}

					if(y>0)
					{
						if(type[(y-1)*(w+2*ex)+x]>1)
						{
							BGRA0=dst.at<Vec4b>(y-1,x);

							if(BGRA0!=Vec4f(255,0,255)&&BGRA1!=Vec4f(255,0,255))
								gy.at<Vec4f>(y,x)=BGRA1-BGRA0;
						}
					}
				}

			}


			//matrix
			cusp::csr_matrix<int,float,cusp::host_memory> A(size,size,5*size-w*2-h*2);
			cusp::array1d<float, cusp::host_memory> X0(A.num_rows, 0);
			cusp::array1d<float, cusp::host_memory> B0(A.num_rows, 0);
			cusp::array1d<float, cusp::host_memory> X1(A.num_rows, 0);
			cusp::array1d<float, cusp::host_memory> B1(A.num_rows, 0);
			cusp::array1d<float, cusp::host_memory> X2(A.num_rows, 0);
			cusp::array1d<float, cusp::host_memory> B2(A.num_rows, 0);
		//	cusp::array1d<float, cusp::host_memory> X3(A.num_rows, 0);
		//	cusp::array1d<float, cusp::host_memory> B3(A.num_rows, 0);
			int nNonZeros=0;
			A.row_offsets[0]=0;


			Vec4f BGRA;
			for(int y=0;y<h+ex*2;y++)
				for (int x=0;x<w+ex*2;x++)
				{
					float a[5];
					a[0]=a[1]=a[2]=a[3]=a[4]=0.0f;
					int ii=y*(w+2*ex)+x;

					switch(type[ii])
					{
					case 0://inside
						break;
					case 1://boundary
						a[2]+=1.0f;
						BGRA=dst.at<Vec4b>(y,x);
						B0[index[ii]]+=BGRA[0];
						B1[index[ii]]+=BGRA[1];
						B2[index[ii]]+=BGRA[2];
					//	B3[index[ii]]+=BGRA[3];

					case 2://outside
						if(y-1>=0&&type[ii-(w+2*ex)]>0)
						{
							a[2]+=1.0f;
							a[0]-=1.0f;
							BGRA=gy.at<Vec4f>(y,x);
							B0[index[ii]]+=BGRA[0];
							B1[index[ii]]+=BGRA[1];
							B2[index[ii]]+=BGRA[2];
						//	B3[index[ii]]+=BGRA[3];
						}
						if(x-1>=0&&type[ii-1]>0)
						{
							a[2]+=1.0f;
							a[1]-=1.0f;
							BGRA=gx.at<Vec4f>(y,x);
							B0[index[ii]]+=BGRA[0];
							B1[index[ii]]+=BGRA[1];
							B2[index[ii]]+=BGRA[2];
						//	B3[index[ii]]+=BGRA[3];
						}
						if(x+1<w+2*ex&&type[ii+1]>0)
						{
							a[2]+=1.0f;
							a[3]-=1.0f;
							BGRA=gx.at<Vec4f>(y,x+1);
							B0[index[ii]]-=BGRA[0];
							B1[index[ii]]-=BGRA[1];
							B2[index[ii]]-=BGRA[2];
						//	B3[index[ii]]+=BGRA[3];
						}
						if(y+1<h+2*ex&&type[ii+(w+2*ex)]>0)
						{
							a[2]+=1.0f;
							a[4]-=1.0f;
							BGRA=gy.at<Vec4f>(y+1,x);
							B0[index[ii]]-=BGRA[0];
							B1[index[ii]]-=BGRA[1];
							B2[index[ii]]-=BGRA[2];
						//	B3[index[ii]]+=BGRA[3];
						}


						//put into A
						if(a[0]!=0)
						{
							A.values[nNonZeros]=a[0];
							A.column_indices[nNonZeros]=index[ii-(w+2*ex)];
							nNonZeros++;
						}
						if(a[1]!=0)
						{
							A.values[nNonZeros]=a[1];
							A.column_indices[nNonZeros]=index[ii-1];
							nNonZeros++;
						}
						if(a[2]!=0)
						{
							A.values[nNonZeros]=a[2];
							A.column_indices[nNonZeros]=index[ii];
							nNonZeros++;
						}
						if(a[3]!=0)
						{
							A.values[nNonZeros]=a[3];
							A.column_indices[nNonZeros]=index[ii+1];
							nNonZeros++;
						}
						if(a[4]!=0)
						{
							A.values[nNonZeros]=a[4];
							A.column_indices[nNonZeros]=index[ii+(w+2*ex)];
							nNonZeros++;
						}
						A.row_offsets[index[ii]+1]=nNonZeros;

						break;
					}
				}

				//SOLVER;
				solve(A,B0,X0);
				solve(A,B1,X1);
				solve(A,B2,X2);
			//	solve(A,B3,X3);

				//paste
				#pragma omp parallel for
				for(int y=0;y<h+ex*2;y++)
					for (int x=0;x<w+ex*2;x++)
					{
						int ii=y*(w+2*ex)+x;
						if(type[ii]>0)
						{

							int B=MIN(MAX(X0[index[ii]],0),255);
							int G=MIN(MAX(X1[index[ii]],0),255);
							int R=MIN(MAX(X2[index[ii]],0),255);
							int A=0;
							dst.at<Vec4b>(y,x)=Vec4b(B,G,R,A);
						}
					}

}

////inline functions////
template<class T_in, class T_out>
T_out CPoissonExt::BilineaGetColor_clamp(cv::Mat& img, float px,float py)//clamp for outside of the boundary
{
	int x[2],y[2];
	T_out value[2][2];
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
			value[i][j]=(img.at<T_in>(temp_y,temp_x));
		}


		return
			value[0][0]*(1-u)*(1-v)+value[0][1]*(1-u)*v+value[1][0]*u*(1-v)+value[1][1]*u*v;
}

