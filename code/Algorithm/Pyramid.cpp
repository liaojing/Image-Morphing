#include "Pyramid.h"
#include "extension.h"
#include "kernel.h"
#include "discrete.h"
#include "error.h"
#include "scale.h"

CPyramids::CPyramids(void)
{
	_gpu=NULL;
}


CPyramids::~CPyramids(void)
{
	levels.clear();
	if (_gpu)
		delete _gpu;
	
}

void  CPyramids::build_pyramid(cv::Mat &image1, cv::Mat &image2,Parameters& para, int n,int order, bool gpu_flag)
{
	if (_gpu)
	delete _gpu;

	if (gpu_flag)
	_gpu=new Morph(para);
	

	_order=order;
	_vector=Mat::zeros(image1.rows,image1.cols,CV_32FC3);
	_qpath=Mat::zeros(image1.rows,image1.cols,CV_32FC3);
	_error=Mat::zeros(image1.rows,image1.cols,CV_32FC1);


	//get the start position of mask
	for(int y=0;y<para.mask1.rows;y++)
	  for(int x=0;x<para.mask1.cols;x++)
		{
			Vec3b bgr=para.mask1.at<Vec3b>(y,x);
			if(bgr[1]==0)
			{
				_p1=QPoint(x,y);
				y=para.mask1.rows;
				x=para.mask1.cols;
			}
		}


		for(int y=0;y<para.mask2.rows;y++)
			for(int x=0;x<para.mask2.cols;x++)
			{
				Vec3b bgr=para.mask2.at<Vec3b>(y,x);
				if(bgr[1]==0)
				{
					_p2=QPoint(x,y);
					y=para.mask2.rows;
					x=para.mask2.cols;
				}
			}

	cv::Mat extends1=cv::Mat(image1.rows,image1.cols,CV_8UC4);
	cv::Mat extends2=cv::Mat(image2.rows,image2.cols,CV_8UC4);
	int from_to[] = { 0,0,1,1,2,2,4,3 };
	cv::Mat src1[2]={image1,para.mask1};
	cv::Mat src2[2]={image2,para.mask2};
	cv::mixChannels(src1, 2, &extends1, 1, from_to, 4 );
	cv::mixChannels(src2, 2, &extends2, 1, from_to, 4 );
	int ex=image1.rows*0.2;
	_extends1=cv::Mat(image1.rows+ex*2,image1.cols+ex*2,CV_8UC4,Scalar(255,255,255,255));
	_extends2=cv::Mat(image2.rows+ex*2,image2.cols+ex*2,CV_8UC4,Scalar(255,255,255,255));
	extends1.copyTo(_extends1(Rect(ex, ex, image1.cols, image1.rows)));
	extends2.copyTo(_extends2(Rect(ex, ex, image2.cols, image2.rows)));

	// use cardinal bspline3 prefilter for downsampling
	kernel::base *pre = new kernel::generalized(
		new kernel::discrete::delta,
		new kernel::discrete::sampled(new kernel::generating::bspline3),
		new kernel::generating::bspline3);
	// no additional discrete processing
	kernel::discrete::base *delta = new kernel::discrete::delta;
	// use mirror extension
	extension::base *ext = new extension::mirror;
	image::rgba<float> rgba1;
	image::load(image1, &rgba1);
	image::rgba<float> rgba2;
	image::load(image2, &rgba2);

	//for multylayers
	#pragma omp parallel for
	for(int y=0;y<para.mask1.rows;y++)
		for(int x=0;x<para.mask1.cols;x++)
		{
			Vec3b bgr1=para.mask1.at<Vec3b>(y,x);
			Vec3b bgr2=para.mask2.at<Vec3b>(y,x);
			if(bgr1[0]==0&&bgr1[1]>0)
			{
				int index=y*para.mask1.cols+x;
				rgba1.r[index]=1.0;
				rgba1.g[index]=1.0;
				rgba1.b[index]=1.0;
			}
			if(bgr2[0]==0&&bgr2[1]>0)
			{
				int index=y*para.mask1.cols+x;
				rgba2.r[index]=1.0;
				rgba2.g[index]=1.0;
				rgba2.b[index]=1.0;
			}
		}

	int w=image1.cols;
	int h=image1.rows;
	CPyramid level;
	cvtColor(image1, level.image1, CV_BGR2GRAY);
	cvtColor(image2, level.image2, CV_BGR2GRAY);
	cv::Mat mask1_one=cv::Mat(h,w,CV_8UC1);
	cv::Mat mask2_one=cv::Mat(h,w,CV_8UC1);
	int fromto[] = { 0,0 };
	cv::mixChannels(&para.mask1, 1, &mask1_one, 1, fromto, 1 );
	cv::mixChannels(&para.mask2, 1, &mask2_one, 1, fromto, 1 );
	level.mask1=mask1_one.clone();
	level.mask2=mask2_one.clone();

	level.blocks_row=((w+4)/5+3)/4*4;
	level.blocks_row+=1-(level.blocks_row%2);//make it odd
	level.blocks_col=(h+4)/5;
	level.blocks_num=level.blocks_row*level.blocks_col;
	level.w=w;
	level.h=h;
	level.inverse_wh=1.0f/(level.w*level.h);

	levels.append(level);

	for(int el=1;el<n;el++)
	{
		w=floor(w/2.0f);
		h=floor(h/2.0f);

		level.image1=Mat::zeros(h,w,CV_8UC3);
		level.image2=Mat::zeros(h,w,CV_8UC3);
		scale(h, w, pre, delta, delta, ext, &rgba1, level.image1);
		scale(h, w, pre, delta, delta, ext, &rgba2, level.image2);

		cvtColor(level.image1, level.image1, CV_BGR2GRAY);
		cvtColor(level.image2, level.image2, CV_BGR2GRAY);

		cv::resize(mask1_one,level.mask1,Size(w,h),0,0,INTER_LINEAR);
		cv::resize(mask2_one,level.mask2,Size(w,h),0,0,INTER_LINEAR);

		level.blocks_row=((w+4)/5+3)/4*4;
		level.blocks_row+=1-(level.blocks_row%2);//make it odd
		level.blocks_col=(h+4)/5;
		level.blocks_num=level.blocks_row*level.blocks_col;
		level.w=w;
		level.h=h;
		level.inverse_wh=1.0f/(level.w*level.h);

		levels.append(level);



	}
}


void  CPyramids::build_pyramid(cv::Mat& mask1, cv::Mat& mask2)
{
	int n=levels.count();

	for(int el=0;el<n;el++)
	{
		//load mask and image
		int w=levels[el].mask1.cols;
		int h=levels[el].mask1.rows;
		cv::resize(mask1,levels[el].mask1,Size(w,h),0,0,INTER_LINEAR);
		cv::resize(mask2,levels[el].mask2,Size(w,h),0,0,INTER_LINEAR);
	}
}
