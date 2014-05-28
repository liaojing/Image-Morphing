#pragma once
#include "../Header.h"
#include "../Algorithm/Pyramid.h"
#ifndef HalfwayImage_H
#define HalfwayImage_H

class HalfwayImage : public QWidget
{
	 Q_OBJECT

public:
	HalfwayImage(char name);
	void setImage(CPyramids* pyramids,int first,int last);
	void setImage(QImage &img1,QImage &img2);
	void setParameters(Parameters* in_parameters);
	const QImage& getImage() const { return _image; }
	template<class T_in, class T_out>
	inline T_out BilineaGetColor_clamp(cv::Mat& img, float px,float py);//clamp for outside of the boundary


protected:
	void mousePressEvent(QMouseEvent *event);
	void mouseMoveEvent(QMouseEvent *event);
	void mouseReleaseEvent(QMouseEvent *event);
	void paintEvent(QPaintEvent *event);
	QSize sizeHint() const;

public:
signals:
	void sigUpdate();
	void sigModified();


public:
	QImage _image;
	QImage _imageL;
	QImage _imageR;
	QImage _image_error;
	bool _image_loaded;
	bool _flag_error;
	bool _flag_multilayer;
	bool _scissor;
	char _name;


private:
	QAction *_pAction;
	QSize _real_size;
	bool _button_up;
	QPointF _mouse_pos;
	bool _pressed;
	Parameters* parameters;

};

#endif

