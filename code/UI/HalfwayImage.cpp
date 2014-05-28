#include "HalfwayImage.h"

HalfwayImage::HalfwayImage(char name)
{
	setAttribute(Qt::WA_StaticContents);
	setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
	setMouseTracking(true);
	_image = QImage(512, 512, QImage::Format_ARGB32);	
	_image.fill(qRgba(240, 240, 240, 255));
	_name=name;
	
	_real_size=_image.size();

	_image_loaded=false;
	_flag_error=false;
	_flag_multilayer=false;
	_pressed=false;
	 _scissor=false;

	_pAction = new QAction(this);
	_pAction->setCheckable(true);
	connect(_pAction, SIGNAL(triggered()), this, SLOT(show()));
	connect(_pAction, SIGNAL(triggered()), this, SLOT(setFocus()));
		
}


QSize HalfwayImage::sizeHint() const
{
	QSize size =  _image.size();
	return size;
}

void HalfwayImage::setImage(QImage &img1,QImage &img2)
{
	_imageL=QImage(img1);
	_imageR=QImage(img2);	

	_image_loaded=true;		

}

void HalfwayImage::setImage(CPyramids* pyramids,int first,int last)
{
	_image=QImage(QSize(_imageL.width(),_imageL.height()), QImage::Format_ARGB32);
  	_image_error=QImage(QSize(_imageL.width(),_imageL.height()), QImage::Format_ARGB32);
	int ex=(pyramids[0]._extends1.rows-pyramids[0]._vector.rows)/2;

	#pragma omp parallel for
	for(int y=0;y<_image.height();y++)
		for(int x=0;x<_image.width();x++)
		{
			Vec4f rgba[MAX_LAYER];
			bool flag[MAX_LAYER];
			for (int i=0;i<MAX_LAYER;i++)
				flag[i]=false;
			
			for (int i=first;i<=last;i++)
			{
				Vec4f rgba1;
				Vec4f rgba2; 
				Vec3f v=pyramids[i]._vector.at<Vec3f>(y,x);	
				rgba1=BilineaGetColor_clamp<Vec4b,Vec4f>(pyramids[i]._extends1,x-v[0]+ex,y-v[1]+ex);
				rgba2=BilineaGetColor_clamp<Vec4b,Vec4f>(pyramids[i]._extends2,x+v[0]+ex,y+v[1]+ex);
				rgba[pyramids[i]._order]=(rgba1+rgba2)/2;
				flag[pyramids[i]._order]=true;
			}

			Vec4f result(255,255,255,255);
			for (int i=0;i<MAX_LAYER;i++)
			{
				if(flag[i])
				{
					float alpha=1.0-rgba[i][3]/255.0f;
					result=result*(1.0-alpha)+rgba[i]*alpha;
				}				
			}
			_image.setPixel(QPoint(x,y),qRgba(result[2],result[1],result[0],255));
						
			if(first==last)
			{
				float error_v=MIN(MAX(pyramids[first]._error.at<float>(y,x),0),1.0)*255;
				_image_error.setPixel(QPoint(x,y),qRgba(error_v,error_v,error_v,255));
			}
		}

		update();		
}


void HalfwayImage::setParameters(Parameters* in_parameters)
{
	parameters=in_parameters;
}

void HalfwayImage::mousePressEvent(QMouseEvent *event)
{
	if (_scissor||_flag_error||_flag_multilayer||!_image_loaded||event->pos().x()>_real_size.width()-1||event->pos().y()>=_real_size.height()-1)
		return;

	if (event->button() == Qt::LeftButton) 
	{
		double2 p;
		p.x=((float)event->pos().x()+0.5)/(float)_real_size.width();
		p.y=((float)event->pos().y()+0.5)/(float)_real_size.height();
		ConstraintPoint elem;		
		elem.lp=elem.rp=p;
		parameters->ui_points.push_back(elem);
		parameters->ActIndex=parameters->ui_points.size()-1;	
		_pressed=true;
	}		

	_mouse_pos=_mouse_pos=QPointF(((float)event->pos().x()+0.5)/(float)_real_size.width(),((float)event->pos().y()+0.5)/(float)_real_size.height());
	emit sigUpdate();
}



void HalfwayImage::mouseMoveEvent(QMouseEvent *event)
{

	if (_scissor||_flag_error||_flag_multilayer||!_image_loaded||event->pos().x()>_real_size.width()-1||event->pos().y()>=_real_size.height()-1)
		return;

	if (event->buttons() == Qt::LeftButton) 
	{
		QPoint pos(MIN(event->pos().x(),_real_size.width()-1),MIN(event->pos().y(),_real_size.height()-1));
		pos=QPoint(MAX(pos.x(),0),MAX(pos.y(),0));

		double2 pointf;
		pointf.x=((float)pos.x()+0.5)/(float)_real_size.width();
		pointf.y=((float)pos.y()+0.5)/(float)_real_size.height();
		parameters->ui_points[parameters->ActIndex].rp=pointf;				
	} 

	_mouse_pos=QPointF(((float)event->pos().x()+0.5)/(float)_real_size.width(),((float)event->pos().y()+0.5)/(float)_real_size.height());
	emit sigUpdate();
}

void HalfwayImage::mouseReleaseEvent(QMouseEvent *event)
{
	if (_scissor||_flag_error||_flag_multilayer||!_image_loaded)
		return;

	if (_pressed) 
	{
		_pressed=false;
		_mouse_pos=_mouse_pos=QPointF(((float)event->pos().x()+0.5)/(float)_real_size.width(),((float)event->pos().y()+0.5)/(float)_real_size.height());
		emit sigModified();		
	}

}



void HalfwayImage::paintEvent(QPaintEvent *event)
{
	if(!_image_loaded)
		return;

	QPainter painter(this);
	QPixmap pixmaptoshow;	
	
		
	if(_flag_error)
		pixmaptoshow=QPixmap::fromImage(_image_error.scaled(this->size(),Qt::KeepAspectRatio));
	else
	{
		QImage tempImage(_image);

		if (!_flag_multilayer)
		{
			QPoint MouseP(_mouse_pos.x()*_image.width(),_mouse_pos.y()*_image.height());
			int radius;
			if (_pressed)		
				radius=_image.width()/8;
			else
				radius=_image.width()/16;
			QRect rect(MouseP-QPoint(radius,radius),MouseP+QPoint(radius,radius));

			for(int y=rect.top();y<=rect.bottom();y++)
				for(int x=rect.left();x<=rect.right();x++)
				{
					if (tempImage.rect().contains(QPoint(x,y))&&(y-MouseP.y())*(y-MouseP.y())+(x-MouseP.x())*(x-MouseP.x())<radius*radius)
					{
						if (_pressed)					
							tempImage.setPixel(QPoint(x,y),_imageR.pixel(QPoint(x,y)));					
						else					
							tempImage.setPixel(QPoint(x,y),_imageL.pixel(QPoint(x,y)));	
					}
				}

				QPainter img_painter(&tempImage);			
				QPen blackPen(qRgba(0, 0, 0, 255));
				img_painter.setPen(blackPen);
				QBrush EmptyBrush(Qt::NoBrush);
				img_painter.setBrush(EmptyBrush);
				img_painter.drawEllipse(MouseP,radius,radius);
		}
		
		pixmaptoshow=QPixmap::fromImage(tempImage.scaled(this->size(),Qt::KeepAspectRatio));
	}

		
	painter.drawPixmap(0,0, pixmaptoshow);
	_real_size=pixmaptoshow.size();		
}


////inline functions////
template<class T_in, class T_out>
T_out HalfwayImage::BilineaGetColor_clamp(cv::Mat& img, float px,float py)//clamp for outside of the boundary
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
