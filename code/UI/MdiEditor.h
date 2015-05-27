#pragma once
#ifndef MdiEditor_H
#define MdiEditor_H


#include "../Header.h"
#include "ImageEditor.h"
#include "RenderWidget.h"
#include "HalfwayImage.h"
#include "DlgPara.h"
#include "../Algorithm/MatchingThread.h"
#include "../Algorithm/MatchingThread_GPU.h"
#include "../Algorithm/QuadraticPath.h"
#include "../Algorithm/PoissonExt.h"
#include "ExternalThread.h"
#include "CtrBar.h"


class MdiEditor : public QMainWindow
{
	Q_OBJECT

public:
	MdiEditor(QApplication* app, QWidget *parent = 0);
	~MdiEditor();
	void contextMenuEvent(QContextMenuEvent* e);

protected:
	bool ReadXmlFile(QString filename);
	bool WriteXmlFile(QString filename);
	void createDockWidget();
	void createStatusBar();
	void createMenuBar();
	void clear();
	void resizeEvent ();
	void Changelayer();
	void DeleteThread(int i);
	void paintEvent(QPaintEvent *event);

public slots:
	void NewProject(bool flag=false);
 	void SaveProject();
	void SaveVector();
	void ModifyPara();
	void updateALL();
	void thread_start(int layer_index);
	void thread_finished(int index);
	void poisson_finished(int index);
	void qpath_finished(int index);
	void SetResults(int index);
	void PtModified();
	void ShowMask();
	void ModifyMask();
	void ModifyLayer();
	void ReorderLayer(char name, bool up,float x, float y);
	void HideMask();
	void ShowHalfway();
	void ShowError();
	void ColorFromImage1();
	void ColorFromImage12();
	void ColorFromImage2();
	void Cancel();
	void Confirm();
	void Layer0();
	void Layer1();
	void Layer2();
	void Layer3();
	void Layer4();
	void MultyLayer();
	void AutoQuit();
	bool CudaInit();
	void CPU_Alg();
	void GPU_Alg();
private:
	QLabel *readyLabel;
	ImageEditor *imageEditorL;
	HalfwayImage *imageEditorM;
	ImageEditor *imageEditorR;
	RenderWidget *imageEditorA;
	QWidget *widgetA;
	CCtrBar *ctrbar;
	QGridLayout *gridLayout;
	QDockWidget *imageDockEditorL;
	QDockWidget *imageDockEditorM;
	QDockWidget *imageDockEditorR;
	QDockWidget *imageDockEditorA;
	QAction *new_pro, *save_pro, *save_vector, *mod_para, *mod_mask, *mod_layer, *show_mask, *hide_mask, *show_halfway, *show_error, *show_image1, *show_image12, *show_image2, *show_multilayer, *show_layer[MAX_LAYER], *gpu_alg, *cpu_alg;
	QAction *cancel,*confirm;
	QMenu *pro_menu,*setting_menu,*view_menu,*alg_menu;
	QMenu *result_view,*color_view,*layer_view,*mask_view;
	cv::Mat image1,image2;
	bool gpu_flag;
	bool gpu_cap;
	//Layers
	int mod;//0-before loaded,1-optimizing,2-new mask, 3-new layer, 4-multilayer
	int layer_num;
	int layer_index;
	CMatchingThread *match_thread[MAX_LAYER];
	CMatchingThread_GPU *match_thread_GPU[MAX_LAYER];
	CPoissonExt *poison_thread[MAX_LAYER];
	CQuadraticPath *qpath_thread[MAX_LAYER];
	Parameters parameters[MAX_LAYER];
	CPyramids pyramids[MAX_LAYER];
	int thread_flag[MAX_LAYER];

public:
	bool _auto;
	QString pro_path;
	QApplication* _app;

}; // class MdiEditor

#endif
