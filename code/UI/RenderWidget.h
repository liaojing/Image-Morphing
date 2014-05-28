#pragma once
#include "../Header.h"
#include "../Algorithm/Pyramid.h"

class RenderWidget:public QGLWidget
{
	  Q_OBJECT
public:
	int w,h,ex;
	int _frame;
	int _minf,_maxf;
	bool _add;
	int _colorfrom;
	GLhandleARB v,p,f;
	uint tex_vector[MAX_LAYER],tex_quadratic[MAX_LAYER],tex1[MAX_LAYER],tex2[MAX_LAYER],tex_fbo;
	GLuint fb;
	bool _loaded;
	bool _save;
	int _index[MAX_LAYER];
	bool _flag[MAX_LAYER];
	Parameters *_parameters;

	QString _pro_path;
	QTimer *_timer;
	float _runtime;
	int _count;

public:
signals:
	void sigRecordFinished();

public:
	RenderWidget();
	~RenderWidget(void);
	void set(QString pro_path, CPyramids* pyramids,int first,int last,Parameters *parameters);

public slots:
	void newframe();
	void RangeChange(int range);
	void StatusChange(int status);

protected:
	void initializeGL();
	void paintGL();
	void keyPressEvent();
	void setShaders();
	char *file2string(const char *path);
	inline float SmoothStep(float t, float a, float b); 

};
