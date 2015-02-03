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
#include <QGLWidget>
#include <QMenu>
#include <QMenuBar>


//opencv
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/types_c.h>  
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

