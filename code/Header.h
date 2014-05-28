#pragma once

//QT
#include <QWidget>
#include <QtGui>
#include <QApplication>
#include <QMainWindow>
#include <QFile>
#include <QTextStream>
#include <QString>
#include <QTimer>
#include <QPoint>
#include <QThread>
#include <QDockWidget>
#include <QMdiArea>
#include <QToolBar>
#include <QAction>
#include <QStatusBar>
#include <QMouseEvent>
#include <QMessageBox>
#include <QDir>
#include <QFileDialog>
#include <QtXml>


//opengl
#if _WIN32
#   include <glew.h>
#   include <glut.h>
#else
#   include <GL/glew.h>
#   include <GL/glut.h>
#endif
#include <Qt/qgl.h>

//opencv
#include <opencv2/opencv.hpp>

using namespace cv;

//openmp
#include <omp.h>

//cuda
#include "CUDA/cg.cuh"

//mkl
#include "mkl.h"

#include "time.h"

//struct
#include <string>
#include <vector>
#include <vector_types.h>

//parameters
#include "parameters.h"

#define MAX_LAYER 5

