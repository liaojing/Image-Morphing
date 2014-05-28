#pragma  once
#ifndef IMAGEEDITOR_H
#define IMAGEEDITOR_H

#include "../Header.h"
#include "../IntelligentScissor/CostGraph.h"
#include "../Algorithm/Pyramid.h"

class ImageEditor : public QWidget
{
     Q_OBJECT

public:
    ImageEditor(char name);
	~ImageEditor();

public:
	QSize sizeHint() const;
    void setImage(cv::Mat& image);
	void setMultiLayer(CPyramids* pyramids,int layer_num);
	void setParameters(Parameters* in_parameters, cv::Mat& mask);
    const QImage& getImage() const { return _image; }
	void clear();

protected:
    void mousePressEvent(QMouseEvent *event);
    void mouseMoveEvent(QMouseEvent *event);
	void paintEvent(QPaintEvent *event);
	void mouseReleaseEvent(QMouseEvent *event);

public:
signals:
	void sigUpdate();
	void sigModified();
	void sigLayerReorder(char name, bool up,float x, float y);//1-up,-1-down


public:
	QImage _image;
	QImage _image_mask;
	QImage _image_layer;

	bool _image_loaded;
	bool _flag_mask;
	bool _flag_multilayer;
	char _name;

	//intelligent scissors
	bool _scissor;
	bool _compute;
	CostGraph* _costGraph;
	QList<QPointF> _fixedSeedList;
	QList<QList<QPointF> > _contourList;//store the contour point seed by seed
	QList<QPointF> _segList;

private:
	QAction *_pAction;
	Parameters* parameters;
	QSize _real_size;

};

#endif


