#include "ImageEditor.h"

ImageEditor::ImageEditor(char name)
{
	setAttribute(Qt::WA_StaticContents);
	setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
	setMouseTracking(true);

	_image = QImage(512, 512, QImage::Format_ARGB32);
	_image.fill(qRgba(240, 240, 240, 255));

	_name=name;	
	parameters=NULL;

	_image_loaded=false;
	_flag_mask=false;
	_flag_multilayer=false;
	_scissor=false;
	_real_size=_image.size();
	_costGraph=NULL;
	_compute=false;

	QList<QPointF> contour;
	_contourList.append(contour);

    _pAction = new QAction(this);
    _pAction->setCheckable(true);
    connect(_pAction, SIGNAL(triggered()), this, SLOT(show()));
    connect(_pAction, SIGNAL(triggered()), this, SLOT(setFocus()));

}

ImageEditor::~ImageEditor()
{
	if(_costGraph)
		delete _costGraph;
}

void ImageEditor::clear()
{
	_segList.clear();
	for (int i=0;i<_contourList.size();i++)
		_contourList[i].clear();	
	_contourList.clear();
	_fixedSeedList.clear();
	//new contour
	QList<QPointF> contour;
	_contourList.append(contour);
	_compute=false;	
}

QSize ImageEditor::sizeHint() const
{
    QSize size =  _image.size();
    return size;
}

void ImageEditor::setImage(cv::Mat& image)
{
	if (image.rows>0) {  
		
		_image=QImage(QSize(image.cols,image.rows), QImage::Format_ARGB32);
		
		#pragma omp parallel for
		for(int y=0;y<image.rows;y++)
			for(int x=0;x<image.cols;x++)
			{				
				Vec3b BGR=image.at<Vec3b>(y,x);			
				_image.setPixel(QPoint(x,y),qRgba(BGR[2],BGR[1],BGR[0],255));
			}
			
			_image_loaded=true;
			update();	
			if(_costGraph)
				delete _costGraph;
			_costGraph = new CostGraph(_image);					
	}
} 

void ImageEditor::setMultiLayer(CPyramids* pyramids,int layer_num)
{
	parameters=NULL;
	int ex=(pyramids[0]._extends1.rows-pyramids[0]._vector.rows)/2;
	_image_layer=QImage(QSize(pyramids[0]._vector.cols,pyramids[0]._vector.rows), QImage::Format_ARGB32);
	for (int i=0;i<layer_num;i++)
	{
		int order=pyramids[i]._order;
		float alpha=(float)(order+1)/(float)layer_num;
		cv::Mat* extends;
		if(_name=='L')
			extends=&(pyramids[i]._extends1);
		else
			extends=&(pyramids[i]._extends2);

		#pragma omp parallel for
		for(int y=0;y<_image.height();y++)
			for (int x=0;x<_image.width();x++)
			{
				Vec4b color;
				color=extends->at<Vec4b>(y+ex,x+ex);
								
				if(color[3]==0)
					_image_layer.setPixel(QPoint(x,y),qRgba(color[2]*alpha,color[1]*alpha,color[0]*alpha,255));
			}
		#pragma omp barrier
	}
	
	 QPainter imagePainter(&_image_layer);
	 imagePainter.setPen(Qt::red);
	 imagePainter.setFont(QFont("Arial", 20));
	 for (int i=0;i<layer_num;i++)
	{
		QString str;
		str.sprintf("%d",pyramids[i]._order);
		if(_name=='L')
			imagePainter.drawText(QRect(pyramids[i]._p1.x(),pyramids[i]._p1.y(),pyramids[i]._p1.x()+50,pyramids[i]._p1.y()+50),str);
		else
			imagePainter.drawText(QRect(pyramids[i]._p2.x(),pyramids[i]._p2.y(),pyramids[i]._p2.x()+50,pyramids[i]._p2.y()+50),str);
	}

}

void ImageEditor::setParameters(Parameters* in_parameters, cv::Mat& mask)
{
	parameters=in_parameters;
	if(mask.rows>0)
	{ 
		_image_layer=QImage(QSize(mask.cols,mask.rows), QImage::Format_ARGB32);
		_image_mask=QImage(QSize(mask.cols,mask.rows), QImage::Format_ARGB32);

		#pragma omp parallel for
		for(int y=0;y<mask.rows;y++)
			for(int x=0;x<mask.cols;x++)
			{
				QRgb rgb=_image.pixel(x,y);
				Vec3b mask_value=mask.at<Vec3b>(y,x);
				
				 float alpha=1.0f-mask_value[0]/510.f;
				  _image_mask.setPixel(QPoint(x,y),qRgba(qRed(rgb)*alpha,qGreen(rgb)*alpha,qBlue(rgb)*alpha,255));	

				if(mask_value[1]>0)
				  _image_layer.setPixel(QPoint(x,y),qRgba(255,255,255,255));
				else
					_image_layer.setPixel(QPoint(x,y),qRgba(qRed(rgb),qGreen(rgb),qBlue(rgb),255));
			
			}		
			update();	
	}
}



void ImageEditor::mousePressEvent(QMouseEvent *event)
{
	if (!_image_loaded||event->pos().x()>_real_size.width()-1||event->pos().y()>=_real_size.height()-1)
		return;

	float x=(float)(event->pos().x()+0.5)/(float)_real_size.width();
	float y=(float)(event->pos().y()+0.5)/(float)_real_size.height();
	
	if(_scissor)
	{		
		if (event->button() == Qt::LeftButton)
		{		
			//confirms seg
			QPointF pointf(x,y);	

			if(_compute)
				for(int i=0;i<_segList.size();i++)
					_contourList[_contourList.size()-1].append(_segList[_segList.size()-1-i]);		

			if (_contourList[_contourList.size()-1].size()>0)
			{				
				QPoint pt0=QPoint(_contourList[_contourList.size()-1][0].x()*_real_size.width(),_contourList[_contourList.size()-1][0].y()*_real_size.height());

				if (abs(pt0.x()-event->pos().x())<=3&&abs(pt0.y()-event->pos().y())<=3)
				{
					//new contour
					QList<QPointF> contour;
					_contourList.append(contour);
					_compute=false;	
					return;
				}	

			}

				_fixedSeedList.append(pointf);				
				_compute=true;
				_segList.clear();				
			
				_costGraph->liveWireDP(y*_image.height(),x*_image.width());	
				emit sigUpdate();
		 }		
	}
	else if(_flag_multilayer)
	{
		if (event->button() == Qt::LeftButton) 
		{
			emit sigLayerReorder(_name,true,x,y);
		}
		else if (event->button() == Qt::RightButton) 
		{
			emit sigLayerReorder(_name,false,x,y);
		}
	}
	else
	{
		if (event->button() == Qt::LeftButton) 
		{
			//select			
			bool flag=true;
			for (int i=0;i<parameters->ui_points.size();i++)
			{
				QPoint ConP;
				switch(_name){
			case 'L':
			case 'l':				
				ConP= QPoint(parameters->ui_points[i].lp.x*_real_size.width(),parameters->ui_points[i].lp.y*_real_size.height());
				break;

			case 'R':
			case 'r':
				ConP= QPoint(parameters->ui_points[i].rp.x*_real_size.width(),parameters->ui_points[i].rp.y*_real_size.height());
				break;
				}

				if (abs(ConP.x()-event->pos().x())<=3&&abs(ConP.y()-event->pos().y())<=3)
				{
					flag=false;
					parameters->ActIndex=i;
					break;
				}
			}

			//new point
			if(flag)
			{
				double2 pointf;
				pointf.x=x;
				pointf.y=y;
				ConstraintPoint elem;					
				elem.lp=elem.rp=pointf;
				parameters->ui_points.push_back(elem);
				parameters->ActIndex=parameters->ui_points.size()-1;		
								
			}
		}		

		else if (event->button() == Qt::RightButton) 
		{
			//select
			std::vector<ConstraintPoint>::iterator itr = parameters->ui_points.begin();
			while (itr != parameters->ui_points.end())
			{
				QPoint ConP;
				switch(_name){
				case 'L':
				case 'l':				
					ConP= QPoint(itr->lp.x*_real_size.width(),itr->lp.y*_real_size.height());
					break;

				case 'R':
				case 'r':
					ConP= QPoint(itr->rp.x*_real_size.width(),itr->rp.y*_real_size.height());
					break;
				}

				if (abs(ConP.x()-event->pos().x())<=3&&abs(ConP.y()-event->pos().y())<=3)
				{
					parameters->ui_points.erase(itr);
					parameters->ActIndex=-1;					
					break;
				}	
				itr++;
			}
			
		}

		emit sigUpdate();

	}

}


void ImageEditor::mouseMoveEvent(QMouseEvent *event)
{

	if(_scissor)
	{
		if (_compute)
		{
			int x=(float)(event->pos().x()+0.5)/(float)_real_size.width()*_image.width();
			int y=(float)(event->pos().y()+0.5)/(float)_real_size.height()*_image.height();
			
			if (x > _image.width()-2)
				x = _image.width()-2;
			if (x < 1)
				x = 1;
			if (y > _image.width()-2)
				y =_image.width()-2;
			if (y < 1)
				y = 1;
			
			_costGraph->computePath(y,x);
						
			//new seg
			_segList.clear();
			for (int i = 0; i < _costGraph->_path.size(); i++)
			{
				float segx = (_costGraph->_path[i]->_col+0.5)/_image.width();
				float segy = (_costGraph->_path[i]->_row+0.5)/_image.height();
				_segList.append(QPointF(segx,segy));				
			}
			emit sigUpdate();
		}
	}
	else
	{
		if (event->buttons() & Qt::LeftButton) 
		{
			if (!_image_loaded||parameters->ActIndex<0||parameters->ActIndex>=parameters->ui_points.size())
				return;

			QPoint pos(MIN(event->pos().x(),_real_size.width()-1),MIN(event->pos().y(),_real_size.height()-1));
			pos=QPoint(MAX(pos.x(),0),MAX(pos.y(),0));

			double2 pointf;
			pointf.x=((float)pos.x()+0.5)/(float)_real_size.width();
			pointf.y=((float)pos.y()+0.5)/(float)_real_size.height();

			QPoint ConP;
			switch(_name){
			case 'L':
			case 'l':				
				parameters->ui_points[parameters->ActIndex].lp=pointf;
				break;

			case 'R':
			case 'r':
				parameters->ui_points[parameters->ActIndex].rp=pointf;
				break;
			}
		} 

		emit sigUpdate();
	}
   
}

void ImageEditor::mouseReleaseEvent(QMouseEvent *event)
{
	if (!_image_loaded||event->pos().x()>_real_size.width()-1||event->pos().y()>=_real_size.height()-1)
		return;

	if (!_scissor&&!_flag_multilayer) 
	{
		emit sigModified();		
	}

}

void ImageEditor::paintEvent(QPaintEvent *event)
{
	if (!_image_loaded)
		return;
	
 	QPainter painter(this);
 	QPixmap pixmaptoshow;
	if(!_flag_mask)
 		pixmaptoshow=QPixmap::fromImage(_image_layer.scaled(this->size(),Qt::KeepAspectRatio));
	else
		pixmaptoshow=QPixmap::fromImage(_image_mask.scaled(this->size(),Qt::KeepAspectRatio));
	
 	painter.drawPixmap(0,0, pixmaptoshow);
 	_real_size=pixmaptoshow.size();
 
	
 	//draw point
	if(_scissor)
	{
		QBrush blackBrush(qRgba(0, 0, 0, 255));
		painter.setBrush(blackBrush);
		for (int i=0;i<_contourList.size();i++)
			for(int j=0;j<_contourList[i].size();j+=3)
		{
			QPoint ConP;						
			ConP=QPoint(_contourList[i][j].x()*_real_size.width(),_contourList[i][j].y()*_real_size.height());
			painter.drawEllipse(ConP,1,1);
		}
		for(int i=0;i<_segList.size();i+=3)
		{
			QPoint ConP;						
			ConP=QPoint(_segList[i].x()*_real_size.width(),_segList[i].y()*_real_size.height());
			painter.drawEllipse(ConP,1,1);
		}

		QBrush redBrush(qRgba(255, 0, 0, 255));
		painter.setBrush(redBrush);

		for(int i=0;i<_fixedSeedList.size();i++)
		{
			
			QPoint ConP;						
			ConP=QPoint(_fixedSeedList[i].x()*_real_size.width(),_fixedSeedList[i].y()*_real_size.height());
		
			painter.drawEllipse(ConP,3,3);
		}

	}
	else
	{
		if(parameters)
		{
			for(int i=0;i<parameters->ui_points.size();i++)
			{
				if(i==parameters->ActIndex)
				{
					QBrush redBrush(qRgba(255, 0, 0, 255));
					painter.setBrush(redBrush);
				}
				else
				{
					QBrush yellowBrush(qRgba(255, 255, 0, 255));
					painter.setBrush(yellowBrush);
				}

				QPoint ConP;
				switch(_name)
				{
				case 'L':
				case 'l':				
					ConP=QPoint(parameters->ui_points[i].lp.x*_real_size.width(),parameters->ui_points[i].lp.y*_real_size.height());
					break;

				case 'R':
				case 'r':
					ConP=QPoint(parameters->ui_points[i].rp.x*_real_size.width(),parameters->ui_points[i].rp.y*_real_size.height());

				}
				painter.drawEllipse(ConP,3,3);
			}

		}
	}	
	
}


 