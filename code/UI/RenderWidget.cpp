#include "GL\glew.h"
#include "GL\glut.h"
#include "RenderWidget.h"
#include "ExternalThread.h"

RenderWidget::RenderWidget()
{
 	setAttribute(Qt::WA_StaticContents);
 	//setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
	_loaded=false;
	_add=true;
	_save=false;

	_colorfrom=1;
	_frame=0;
	_maxf=100;
	_minf=0;
	_runtime=0.0f;
	_count=0;
 	_timer=new QTimer(this);
 	connect(_timer,SIGNAL(timeout()), this, SLOT(newframe()) );
 	_timer->start(100);
}

void RenderWidget::set(QString pro_path, CPyramids* pyramids,int first,int last,Parameters* parameters)
{
	if(_loaded)
	{
		glDeleteTextures(MAX_LAYER,tex_vector);
		glDeleteTextures(MAX_LAYER,tex_quadratic);
		glDeleteTextures(MAX_LAYER,tex1);
		glDeleteTextures(MAX_LAYER,tex2);
		glDeleteFramebuffersEXT(1, &fb);
		glDeleteTextures(1,&tex_fbo);
	}

	w=pyramids[0]._vector.cols;
	h=pyramids[0]._vector.rows;
	ex=(pyramids[0]._extends1.cols-w)/2;

	for (int i=0;i<MAX_LAYER;i++)
		_flag[i]=false;

	glEnable(GL_TEXTURE_2D);
	glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);
	glGenTextures(MAX_LAYER, tex_vector);
	glGenTextures(MAX_LAYER, tex_quadratic);
	glGenTextures(MAX_LAYER, tex1);
	glGenTextures(MAX_LAYER, tex2);

	for(int i=first;i<=last;i++)
	{
		_flag[pyramids[i]._order]=true;
		_index[pyramids[i]._order]=i;
		//2D RGBA32 float
		glBindTexture(GL_TEXTURE_2D, tex_vector[i]);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB32F, w, h, 0, GL_RGB, GL_FLOAT, pyramids[i]._vector.data);

		//2D RGBA32 float

		glBindTexture(GL_TEXTURE_2D, tex_quadratic[i]);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB32F, w, h, 0, GL_RGB, GL_FLOAT, pyramids[i]._qpath.data);


		//image1

		glBindTexture(GL_TEXTURE_2D, tex1[i]);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, w+ex*2, h+ex*2, 0, GL_BGRA, GL_UNSIGNED_BYTE, pyramids[i]._extends1.data);

		//image2
		glBindTexture(GL_TEXTURE_2D, tex2[i]);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, w+ex*2, h+ex*2, 0, GL_BGRA, GL_UNSIGNED_BYTE, pyramids[i]._extends2.data);

	}

	//FBO
	glGenFramebuffersEXT(1, &fb);// frame buffer
	glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, fb);

	glGenTextures(1, &tex_fbo);// texture
	glBindTexture(GL_TEXTURE_2D, tex_fbo);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, w, h, 0,GL_RGBA, GL_UNSIGNED_BYTE, NULL);
	glFramebufferTexture2DEXT(GL_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT0_EXT, GL_TEXTURE_2D, tex_fbo, 0);

	 _loaded=true;
	_pro_path=pro_path;
	_parameters=parameters;


}

void RenderWidget::newframe()
{
	if(_minf>=_maxf)
		{
			_frame=_minf;
			return;
	}

	if(_add)
	{
		_frame++;
		if(_frame>=_maxf)
			_add=false;

	}
	else
	{
		_frame--;
		if(_frame<=_minf)
			_add=true;

	}

	update();


}

RenderWidget::~RenderWidget(void)
{
	glDetachObjectARB(p,v);
	glDetachObjectARB(p,f);

	glDeleteObjectARB(v);
	glDeleteObjectARB(f);
	glDeleteObjectARB(p);

	if(_loaded)
	{
		glDeleteTextures(MAX_LAYER,tex_vector);
		glDeleteTextures(MAX_LAYER,tex_quadratic);
		glDeleteTextures(MAX_LAYER,tex1);
		glDeleteTextures(MAX_LAYER,tex2);
		glDeleteFramebuffersEXT(1, &fb);
		glDeleteTextures(1,&tex_fbo);
	}

	delete _timer;
}

void RenderWidget::initializeGL()
{
	glShadeModel( GL_SMOOTH );
	glDisable( GL_DEPTH_TEST );
	glDisable(GL_CULL_FACE);
	glEnable(GL_TEXTURE_2D);
	glEnable(GL_POINT_SMOOTH);
	glEnable(GL_LINE_SMOOTH);
	glHint( GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST );
	glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
	glPixelStorei(GL_PACK_ALIGNMENT, 1);
	glewInit();
	setShaders();

}

void RenderWidget::setShaders()
{
	char *fs;
	//v = glCreateShaderObjectARB(GL_VERTEX_SHADER_ARB);
	f = glCreateShaderObjectARB(GL_FRAGMENT_SHADER_ARB);
	//vs = file2string("vertexshader.txt");
	fs = file2string("fragshader.txt");
	//const char * vv = vs;
	const char * ff = fs;
	//glShaderSourceARB(v, 1, &vv,NULL);
	glShaderSourceARB(f, 1, &ff,NULL);
	//free(vs);
	free(fs);
	//glCompileShaderARB(v);
	glCompileShaderARB(f);
	/*GLint isCompiled = 0;
	glGetShaderiv(f, GL_COMPILE_STATUS, &isCompiled);
	if(isCompiled == GL_FALSE)
	{
		GLint maxLength = 0;
		glGetShaderiv(f, GL_INFO_LOG_LENGTH, &maxLength);

		//The maxLength includes the NULL character
		char* infoLog=new char[maxLength];
		glGetShaderInfoLog(f, maxLength, &maxLength, infoLog);

		//We don't need the shader anymore.
		glDeleteShader(f);

		//Use the infoLog as you see fit.
		QFile file("error.txt");							
		if (file.open(QFile::WriteOnly | QFile::Truncate))
		{
			QTextStream out(&file);	
			QString line;
			//mp4_1
			line.sprintf("%s",infoLog);
			out<<line;
				
		}
		file.flush(); 
		file.close(); 
		delete[] infoLog;
		//In this simple program, we'll just leave
		return;
	}*/
	p = glCreateProgramObjectARB();
	//glAttachObjectARB(p,v);
	glAttachObjectARB(p,f);
	glLinkProgramARB(p);

}

char* RenderWidget::file2string(const char *path)
{
	FILE *fd;
	long len,
		r;
	char *str;

	if (!(fd = fopen(path, "r")))
	{
		fprintf(stderr, "Can't open file '%s' for reading\n", path);
		QMessageBox::critical(NULL, "critical", "Please put fragshader.txt in the same directory of exe", QMessageBox::Yes);
		return NULL;
	}

	fseek(fd, 0, SEEK_END);
	len = ftell(fd);

	printf("File '%s' is %ld long\n", path, len);

	fseek(fd, 0, SEEK_SET);

	if (!(str =(char*) malloc(len * sizeof(char))))
	{
		fprintf(stderr, "Can't malloc space for '%s'\n", path);
		return NULL;
	}

	r = fread(str, sizeof(char), len, fd);

	str[r - 1] = '\0'; /* Shader sources have to term with null */

	fclose(fd);

	return str;
}

void RenderWidget::paintGL()
{
	if(_loaded)
	{
		GLuint queries[1];
		GLuint queries_result;
		glGenQueries(1, queries);
		glBeginQuery(GL_TIME_ELAPSED, queries[0]);
		//first pass
		glEnable(GL_BLEND);
		glBlendFunc(GL_ONE_MINUS_SRC_ALPHA,GL_SRC_ALPHA);
		glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, fb);
		glUseProgramObjectARB(p);
		glClearColor(1.0,1.0,1.0,0.0);
		glClear( GL_COLOR_BUFFER_BIT);
		glLoadIdentity();

		glViewport( 0, 0, w, h);
		glMatrixMode( GL_PROJECTION );
		glLoadIdentity();
		gluOrtho2D(0,w,0,h);
		glMatrixMode( GL_MODELVIEW );
		glLoadIdentity();		

		float t=(float)_frame/100.0f;
		float t_geo=SmoothStep(t,0.0f,1.0f);
		float t_color=SmoothStep(t,0.2f,0.8f);

		float loc=glGetUniformLocationARB(p,"t_geo");
		glUniform1fARB(loc,t_geo);
		loc=glGetUniformLocationARB(p,"t_color");
		glUniform1fARB(loc,t_color);
		loc=glGetUniformLocationARB(p,"ex");
		glUniform1fARB(loc,(float)ex);
		loc=glGetUniformLocationARB(p,"w");
		glUniform1fARB(loc,(float)w);
		loc=glGetUniformLocationARB(p,"h");
		glUniform1fARB(loc,(float)h);
		loc=glGetUniformLocationARB(p,"color_from");
		glUniform1iARB(loc,_colorfrom);

		float tex_loc=glGetUniformLocationARB(p,"vector");
		glUniform1iARB(tex_loc,0);
		tex_loc=glGetUniformLocationARB(p,"quadratic");
		glUniform1iARB(tex_loc,1);
		tex_loc=glGetUniformLocationARB(p,"image1");
		glUniform1iARB(tex_loc,2);
		tex_loc=glGetUniformLocationARB(p,"image2");
		glUniform1iARB(tex_loc,3);

		for(int i=0;i<MAX_LAYER;i++)
		{
			if(_flag[i])
			{
				glActiveTextureARB(GL_TEXTURE0);
				glBindTexture(GL_TEXTURE_2D,tex_vector[_index[i]]);
				glActiveTextureARB(GL_TEXTURE1);
				glBindTexture(GL_TEXTURE_2D,tex_quadratic[_index[i]]);
				glActiveTextureARB(GL_TEXTURE2);
				glBindTexture(GL_TEXTURE_2D,tex1[_index[i]]);
				glActiveTextureARB(GL_TEXTURE3);
				glBindTexture(GL_TEXTURE_2D,tex2[_index[i]]);

				glBegin(GL_QUADS);
				glVertex2f(0,0);
				glVertex2f(w,0);
				glVertex2f(w,h);
				glVertex2f(0,h);
				glEnd();
			}

		}

		glFlush();

		glEndQuery(GL_TIME_ELAPSED);

		if(_save)
		{
			glGetQueryObjectuiv(queries[0], GL_QUERY_RESULT, &queries_result);
			_runtime+=queries_result*0.000001f;
			_count++;
			glReadBuffer(GL_COLOR_ATTACHMENT0_EXT);
			uchar *buffer=new uchar[w*h*3];
			glReadPixels(0,0,w,h,GL_RGB,GL_UNSIGNED_BYTE,buffer);
			cv::Mat img=Mat::zeros(h,w,CV_8UC3);
			#pragma omp parallel for
			for (int y=0;y<h;y++)
				for (int x=0;x<w;x++)
				{
					int index=(h-1-y)*w*3+x*3;
					img.at<Vec3b>(y,x)=Vec3b(buffer[index+2],buffer[index+1],buffer[index+0]);
				}
			delete[] buffer;
			QString filename;

			int a=_frame/100;
			int b=(_frame%100)/10;
			int c=_frame%10;
			filename.sprintf("%s\\frame%d%d%d.png",_pro_path.toLatin1().data(),a,b,c);

			cv::imwrite(filename.toLatin1().data(),img);

			if(_frame==0||_frame==25||_frame==50||_frame==75||_frame==100)
			{
			if(_frame==0&&_parameters)
			{
			for (int i=0;i<_parameters->ui_points.size();i++)
			{
			int x=_parameters->ui_points[i].lp.x*w-0.5f;
			int y=_parameters->ui_points[i].lp.y*h-0.5f;
			circle(img, Point(x,y), w*0.01, Scalar(0,0,255), -1);
			}
			}
			else if(_frame==100&&_parameters)
			{
			for (int i=0;i<_parameters->ui_points.size();i++)
			{
			int x=_parameters->ui_points[i].rp.x*w-0.5f;
			int y=_parameters->ui_points[i].rp.y*h-0.5f;
			circle(img, Point(x,y), w*0.01, Scalar(0,0,255), -1);
			}
			}

			filename.sprintf("%s\\%d_color%d.png",_pro_path.toLatin1().data(),_frame,_colorfrom);

			cv::imwrite(filename.toLatin1().data(),img);							
			}
			if(_frame>=_maxf)
			{					
				_save=false;			

				QFile file("all.bat");							
				if (file.open(QFile::WriteOnly | QFile::Truncate))
				{
					QTextStream out(&file);	
					QString line;
					//mp4_1
					line.sprintf("libav\\avconv.exe -r 15 -i %s\\frame%%%%03d.png -y %s\\movie_color%d.mp4\n",_pro_path.toLatin1().data(),_pro_path.toLatin1().data(),_colorfrom);
					out<<line;
					//del
 					line.sprintf("del %s\\frame???.png\n",_pro_path.toLatin1().data());
 					out<<line;	

					//del
					line.sprintf("del all.bat\n");
					out << line;

				}
				file.flush(); 
				file.close(); 

					
				CExternalThread* external_thread = new CExternalThread();
				external_thread->start(QThread::HighestPriority);
				external_thread->wait();
				_runtime/=_count;
				emit sigRecordFinished();

			}
		}


		//second pass
		glDisable(GL_BLEND);
		glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, 0);
		glUseProgramObjectARB(0);
		glClearColor(1.0,1.0,1.0,0.0);
		glClear( GL_COLOR_BUFFER_BIT);
		glDisable(GL_TEXTURE_2D);
		glLoadIdentity();

		glActiveTextureARB(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_2D, tex_fbo);

		float ratio_w=this->size().width()/(float)w;
		float ratio_h=this->size().height()/(float)h;
		int real_w,real_h;
		if(ratio_w<ratio_h)
			real_w=w*ratio_w,real_h=h*ratio_w;
		else
			real_w=w*ratio_h,real_h=h*ratio_h;

		glViewport( 0, 0, real_w, real_h);
		glMatrixMode( GL_PROJECTION );
		glLoadIdentity();
		gluOrtho2D(0,real_w,0,real_h);
		glMatrixMode( GL_MODELVIEW );
		glLoadIdentity();

		glBegin(GL_QUADS);
		glTexCoord2f(0.0f,0.0f);
		glVertex2f(0,0);
		glTexCoord2f(1.0f,0.0f);
		glVertex2f(real_w,0);
		glTexCoord2f(1.0f,1.0f);
		glVertex2f(real_w,real_h);
		glTexCoord2f(0.0f,1.0f);
		glVertex2f(0,real_h);
		glEnd();

		glFlush();


	}
	else
	{
		glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, 0);
		glUseProgramObjectARB(0);
		glClearColor(0.9412,0.9412,0.9412,0.0);
		glClear( GL_COLOR_BUFFER_BIT);
	}
}


void RenderWidget::RangeChange(int range)
{
	_timer->stop();
	_minf=50-range;
	_maxf=50+range;
	_frame=_minf;
	_timer->start(100);
	update();
}

void RenderWidget::StatusChange(int status)
{

   switch(status)
   {
	   case 0:
		   _timer->stop();
		   _frame=_minf;
		   _save=true;
		   _add=true;
		   _timer->start(100);
		   break;
	   case 1:
		   _timer->start(100);
		   break;
	   case 2:
		   _timer->stop();
		   break;
	   case 3:
		   _timer->stop();
		   _frame=50;
		   break;

   }
   update();
}

inline float RenderWidget::SmoothStep(float t, float a, float b) {
	if (t < a) return 0;
	if (t > b) return 1;
	t = (t-a)/(b-a);
	return t*t * (3-2*t);
}