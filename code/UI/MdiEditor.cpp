#include "MdiEditor.h"

MdiEditor::MdiEditor(QApplication* app,QWidget *parent)
: QMainWindow(parent)
{

	_app=app;
	readyLabel=NULL;
	new_pro=save_pro=save_vector=mod_para=NULL;
	imageEditorL = new ImageEditor('L');
	imageEditorR = new ImageEditor('R');
	imageEditorM = new HalfwayImage('M');
	widgetA = new QWidget();
	imageEditorA = new RenderWidget();
	ctrbar=new CCtrBar();

	for(int i=0;i<MAX_LAYER;i++)
	{
		match_thread[i]=NULL;
		match_thread_GPU[i]=NULL;
		poison_thread[i]=NULL;
		qpath_thread[i]=NULL;
	}


	createDockWidget();
	createMenuBar();
	createStatusBar();
	gpu_flag=FALSE;
	gpu_cap=CudaInit();

	connect(imageEditorL,SIGNAL(sigUpdate()),this,SLOT(updateALL()));
	connect(imageEditorR,SIGNAL(sigUpdate()),this,SLOT(updateALL()));
	connect(imageEditorM,SIGNAL(sigUpdate()),this,SLOT(updateALL()));
	connect(ctrbar,SIGNAL(sigUpdate()),this,SLOT(updateALL()));
	connect(imageEditorL,SIGNAL(sigModified()),this,SLOT(PtModified()));
	connect(imageEditorR,SIGNAL(sigModified()),this,SLOT(PtModified()));
	connect(imageEditorM,SIGNAL(sigModified()),this,SLOT(PtModified()));
	connect(imageEditorL,SIGNAL(sigLayerReorder(char, bool,float, float)),this,SLOT(ReorderLayer(char, bool,float, float)));
	connect(imageEditorR,SIGNAL(sigLayerReorder(char, bool,float, float)),this,SLOT(ReorderLayer(char, bool,float, float)));
	connect(ctrbar,SIGNAL(sigStatusChange(int)),imageEditorA,SLOT(StatusChange(int)));
	connect(ctrbar,SIGNAL(sigRangeChange(int)),imageEditorA,SLOT(RangeChange(int)));
	connect(imageEditorA,SIGNAL(sigRecordFinished()),ctrbar,SLOT(record_finished()));
	connect(imageEditorA,SIGNAL(sigRecordFinished()),this,SLOT(AutoQuit()));

	layer_num=0;
	layer_index=0;

	clear();
    setWindowTitle(tr("Pixel Morph"));
	showMaximized();

}

bool MdiEditor::CudaInit()
{
	int i;
	int device_count;
	if( cudaGetDeviceCount(&device_count) )
		return false;

	for(i=0;i<device_count;i++)
	{
		struct cudaDeviceProp device_prop;
		if(cudaGetDeviceProperties(&device_prop,i)==cudaSuccess)
		{
			if(device_prop.major>=3)
			{
				if(cudaSetDevice(i)==cudaSuccess)
						return true;
			}
		}
	}
	return false;

}

void MdiEditor::DeleteThread(int i)
{
	if (i<0 || i>layer_num)
		return;

	if (match_thread[i])
	{
		disconnect(match_thread[i],0,0,0);
		match_thread[i]->runflag=false;
		match_thread[i]->wait();
		match_thread[i]->deleteLater();
		match_thread[i]=NULL;
	}

	if (match_thread_GPU[i])
	{
		disconnect(match_thread_GPU[i],0,0,0);
		match_thread_GPU[i]->runflag=false;
		match_thread_GPU[i]->wait();
		match_thread_GPU[i]->deleteLater();
		match_thread_GPU[i]=NULL;
	}

	if(poison_thread[i])
	{
		disconnect(poison_thread[i],0,0,0);
		poison_thread[i]->wait();
		poison_thread[i]->deleteLater();
		poison_thread[i]=NULL;
	}

	if(qpath_thread[i])
	{
		disconnect(qpath_thread[i],0,0,0);
		qpath_thread[i]->wait();
		qpath_thread[i]->deleteLater();
		qpath_thread[i]=NULL;
	}


}
void MdiEditor::clear()
{
	for (int i=0;i<layer_num;i++)
	{
		DeleteThread(i);
		thread_flag[i]=0;

		switch(i)
		{
		case 0:
			disconnect(show_layer[i],SIGNAL(triggered()),this,SLOT(Layer0()));
			break;
		case 1:
			connect(show_layer[i],SIGNAL(triggered()),this,SLOT(Layer1()));
			break;
		case 2:
			connect(show_layer[i],SIGNAL(triggered()),this,SLOT(Layer2()));
			break;
		case 3:
			connect(show_layer[i],SIGNAL(triggered()),this,SLOT(Layer3()));
			break;
		case 4:
			connect(show_layer[i],SIGNAL(triggered()),this,SLOT(Layer4()));
			break;
		}

		layer_view->removeAction(show_layer[i]);
		show_layer[i]->deleteLater();
		show_layer[i]=NULL;

	}

	for(int i=0;i<MAX_LAYER;i++)
	{

		parameters[i].ui_points.clear();
		parameters[i].ActIndex=-1;
		parameters[i].w_ssim=1.0f;
		parameters[i].ssim_clamp=0.0f;
		parameters[i].w_tps=0.001f;
		parameters[i].w_ui=100.0f;
		parameters[i].max_iter=2000;
		parameters[i].max_iter_drop_factor=2;
		parameters[i].eps=0.01f;
		parameters[i].start_res=8;
		parameters[i].bcond=BCOND_NONE;
		parameters[i].verbose=true;
	}
	pyramids->levels.clear();

	imageEditorL->_image_loaded=false;
	imageEditorR->_image_loaded=false;
	imageEditorM->_image_loaded=false;
	imageEditorM->_flag_error=false;
	imageEditorA->_loaded=false;
	imageEditorA->_colorfrom=1;
	ctrbar->_status=-1;

	layer_num=0;
	layer_index=0;
	mod=0;

}

void MdiEditor::paintEvent(QPaintEvent *event)
{
        (void)event; // ignore argument

	switch(mod)
	{
	case 0://before loaded
		pro_menu->setEnabled(true);
		view_menu->setEnabled(false);
		setting_menu->setEnabled(false);
		alg_menu->setEnabled(false);
		break;
	case 1://optimizing
		pro_menu->setEnabled(true);
		view_menu->setEnabled(true);
		setting_menu->setEnabled(true);
		alg_menu->setEnabled(true);
		break;
	case 2://set mask
		pro_menu->setEnabled(false);
		view_menu->setEnabled(false);
		setting_menu->setEnabled(false);
		alg_menu->setEnabled(false);
		break;
	case 3://set layer
		pro_menu->setEnabled(false);
		view_menu->setEnabled(false);
		setting_menu->setEnabled(false);
		alg_menu->setEnabled(false);
		break;
	case 4://multilayer
		pro_menu->setEnabled(true);
		view_menu->setEnabled(true);
		result_view->setEnabled(false);
		mask_view->setEnabled(false);
		setting_menu->setEnabled(false);
		alg_menu->setEnabled(false);
		break;
	}

	if(layer_num>1)
	{
		bool flag=true;
 		for(int i=0;i<layer_num;i++)
 			if (thread_flag[i]!=4)
 				flag=false;
		show_multilayer->setEnabled(flag);
	}
	else
		show_multilayer->setEnabled(false);

	if(layer_num<MAX_LAYER)
		mod_layer->setEnabled(true);
	else
		mod_layer->setEnabled(false);


	for (int i=0;i<layer_num;i++)
		show_layer[i]->setChecked(false);
	show_multilayer->setChecked(false);
	if(layer_num>0&&layer_index>=0)
		show_layer[layer_index]->setChecked(true);
	if(layer_index<0)
		show_multilayer->setChecked(true);

	if(imageEditorL->_flag_mask)
	{
		show_mask->setChecked(true);
		hide_mask->setChecked(false);
	}
	else
	{
		show_mask->setChecked(false);
		hide_mask->setChecked(true);
	}

	switch(imageEditorA->_colorfrom)
	{
		case 0:
			show_image1->setChecked(true);
			show_image12->setChecked(false);
			show_image2->setChecked(false);
			break;
		case 1:
			show_image1->setChecked(false);
			show_image12->setChecked(true);
			show_image2->setChecked(false);
			break;
		case 2:
			show_image1->setChecked(false);
			show_image12->setChecked(false);
			show_image2->setChecked(true);
			break;
	}
	if (imageEditorM->_flag_error)
	{
		show_halfway->setChecked(false);
		show_error->setChecked(true);
	}
	else
	{
		show_halfway->setChecked(true);
		show_error->setChecked(false);
	}

	if(gpu_flag)
	{
		
		gpu_alg->setChecked(true);
		cpu_alg->setChecked(false);
		mod_mask->setEnabled(false);
		mod_layer->setEnabled(false);

	}
	else
	{
		gpu_alg->setChecked(false);
		cpu_alg->setChecked(true);
		mod_mask->setEnabled(true);
		mod_layer->setEnabled(true);
	}

	if (thread_flag[layer_index] < 4 && thread_flag[layer_index]>1)
	{
		gpu_alg->setEnabled(false);
		cpu_alg->setEnabled(false);
	}
	else
	{
		
		gpu_alg->setEnabled(true);
		cpu_alg->setEnabled(true);
	}

	if (!gpu_cap)
		gpu_alg->setEnabled(false);
}

MdiEditor::~MdiEditor()
{
	clear();

	if(readyLabel)
		delete readyLabel;
	if(imageEditorL)
		delete imageEditorL;
	if(imageEditorM)
		delete imageEditorM;
	if(imageEditorR)
		delete imageEditorR;
	if(imageEditorA)
		delete imageEditorA;
	if(ctrbar)
		delete ctrbar;
	if(gridLayout)
		delete gridLayout;
	if(widgetA)
		delete widgetA;
	if(imageDockEditorL)
		delete imageDockEditorL;
	if(imageDockEditorM)
		delete imageDockEditorM;
	if(imageDockEditorR)
		delete imageDockEditorR;
	if(imageDockEditorA)
		delete imageDockEditorA;


}

 void MdiEditor::createDockWidget()
 {
	imageDockEditorL = new QDockWidget(tr("Input image1"),this);
	imageDockEditorL->setFeatures(QDockWidget::DockWidgetMovable|
 		QDockWidget::DockWidgetFloatable);
 	imageDockEditorL->setAllowedAreas(Qt::AllDockWidgetAreas);
 	imageDockEditorL->setWidget(imageEditorL);
	imageDockEditorR = new QDockWidget(tr("Input image2"),this);
 	imageDockEditorR->setFeatures(QDockWidget::DockWidgetMovable|
 		QDockWidget::DockWidgetFloatable);
 	imageDockEditorR->setAllowedAreas(Qt::AllDockWidgetAreas);
 	imageDockEditorR->setWidget(imageEditorR);
 	imageDockEditorM = new QDockWidget(tr("Halfway result"),this);
 	imageDockEditorM->setFeatures(QDockWidget::DockWidgetMovable|
 		QDockWidget::DockWidgetFloatable);
 	imageDockEditorM->setAllowedAreas(Qt::AllDockWidgetAreas);
 	imageDockEditorM->setWidget(imageEditorM);
 	imageDockEditorA = new QDockWidget(tr("Morphing result"),this);
 	imageDockEditorA->setFeatures(QDockWidget::DockWidgetMovable|
 		QDockWidget::DockWidgetFloatable);
 	imageDockEditorA->setAllowedAreas(Qt::AllDockWidgetAreas);

	gridLayout=new QGridLayout();
	gridLayout->addWidget(ctrbar,0,0);
 	gridLayout->addWidget(imageEditorA,1,0,10,1);
	widgetA->setLayout(gridLayout);
	imageDockEditorA->setWidget(widgetA);



 	addDockWidget(Qt::TopDockWidgetArea,imageDockEditorL);
 	addDockWidget(Qt::RightDockWidgetArea,imageDockEditorR);
 	addDockWidget(Qt::LeftDockWidgetArea,imageDockEditorM);
 	addDockWidget(Qt::BottomDockWidgetArea,imageDockEditorA);

 	setCorner(Qt::TopLeftCorner,Qt::TopDockWidgetArea);
 	setCorner(Qt::TopRightCorner,Qt::RightDockWidgetArea);
 	setCorner(Qt::BottomLeftCorner,Qt::LeftDockWidgetArea);
 	setCorner(Qt::BottomRightCorner,Qt::BottomDockWidgetArea);

 }

 void MdiEditor::createStatusBar()
 {
     readyLabel = new QLabel(tr("Please create/load a project"));
     statusBar()->addWidget(readyLabel, 1);
 }

 void MdiEditor::createMenuBar()
 {
	pro_menu=menuBar()->addMenu("Project");
	new_pro=new QAction("&New Project", this);
	save_pro=new QAction("&Save Project", this);
	save_vector = new QAction("&Save Vector", this);
	pro_menu->addAction(new_pro);
	pro_menu->addAction(save_pro);
	pro_menu->addAction(save_vector);

	setting_menu=menuBar()->addMenu("Settings");
	mod_para=new QAction("&Parameters", this);
	setting_menu->addAction(mod_para);
	mod_mask=new QAction("&Masks", this);
	setting_menu->addAction(mod_mask);
	mod_layer=new QAction("&Layers", this);
	setting_menu->addAction(mod_layer);


	view_menu=menuBar()->addMenu("View");
	mask_view=view_menu->addMenu("Mask");
	show_mask=new QAction("&Show", this);
	mask_view->addAction(show_mask);
	hide_mask=new QAction("&Hide", this);
	mask_view->addAction(hide_mask);

	result_view=view_menu->addMenu("Results");
	show_halfway=new QAction("&Halfway Image", this);
	result_view->addAction(show_halfway);
	show_error=new QAction("&Error Image", this);
	result_view->addAction(show_error);

	color_view=view_menu->addMenu("Color from");
	show_image1=new QAction("&Image1", this);
	color_view->addAction(show_image1);
	show_image12=new QAction("&Both image1 & image2", this);
	color_view->addAction(show_image12);
	show_image2=new QAction("&Image2", this);
	color_view->addAction(show_image2);

	layer_view=view_menu->addMenu("Layer");
	show_multilayer=new QAction("&Multilayer", this);
	layer_view->addAction(show_multilayer);

	alg_menu=menuBar()->addMenu("Algorithm");
	cpu_alg=new QAction("&CPU", this);
	gpu_alg=new QAction("&GPU", this);
	alg_menu->addAction(cpu_alg);
	alg_menu->addAction(gpu_alg);


	show_mask->setCheckable(true);
	hide_mask->setCheckable(true);
	show_halfway->setCheckable(true);
	show_error->setCheckable(true);
	show_image1->setCheckable(true);
	show_image12->setCheckable(true);
	show_image2->setCheckable(true);
	show_multilayer->setCheckable(true);
	cpu_alg->setCheckable(true);
	gpu_alg->setCheckable(true);

	//right menu
	confirm=new QAction("&confirm",this);
	cancel=new QAction("&cancel",this);

	//signal
	connect(new_pro,SIGNAL(triggered()),this,SLOT(NewProject()));
	connect(save_pro,SIGNAL(triggered()),this,SLOT(SaveProject()));
	connect(save_vector, SIGNAL(triggered()), this, SLOT(SaveVector()));
	connect(mod_para,SIGNAL(triggered()),this,SLOT(ModifyPara()));
	connect(mod_mask,SIGNAL(triggered()),this,SLOT(ModifyMask()));
	connect(show_mask,SIGNAL(triggered()),this,SLOT(ShowMask()));
	connect(hide_mask,SIGNAL(triggered()),this,SLOT(HideMask()));
	connect(show_halfway,SIGNAL(triggered()),this,SLOT(ShowHalfway()));
	connect(show_error,SIGNAL(triggered()),this,SLOT(ShowError()));
	connect(show_image1,SIGNAL(triggered()),this,SLOT(ColorFromImage1()));
	connect(show_image12,SIGNAL(triggered()),this,SLOT(ColorFromImage12()));
	connect(show_image2,SIGNAL(triggered()),this,SLOT(ColorFromImage2()));
	connect(confirm,SIGNAL(triggered()),this,SLOT(Confirm()));
	connect(cancel,SIGNAL(triggered()),this,SLOT(Cancel()));
	connect(mod_layer,SIGNAL(triggered()),this,SLOT(ModifyLayer()));
	connect(show_multilayer,SIGNAL(triggered()),this,SLOT(MultyLayer()));
	connect(cpu_alg,SIGNAL(triggered()),this,SLOT(CPU_Alg()));
	connect(gpu_alg,SIGNAL(triggered()),this,SLOT(GPU_Alg()));


 }

 void MdiEditor::resizeEvent ()
 {
 	imageEditorM->update();
 	imageEditorM->updateGeometry();
 	imageEditorL->update();
 	imageEditorL->updateGeometry();
 	imageEditorR->update();
 	imageEditorR->updateGeometry();
 }

 void MdiEditor::updateALL()
 {
 	imageEditorM->update();
 	imageEditorL->update();
 	imageEditorR->update();
	imageEditorA->update();
  }

 void MdiEditor::NewProject(bool flag)
 {
 	clear();
	if (!flag)
	  	pro_path = QFileDialog::getExistingDirectory (this);
  	if(!pro_path.isNull())
  	{

  		if(!ReadXmlFile(pro_path+"\\settings.xml"))//exist

  		{

  			QFile::remove(pro_path+"\\result.html");
  			QFile::copy("template.html",pro_path+"\\result.html");


  			//load two images
  			QString ImagePathName = QFileDialog::getOpenFileName(
  				this,
  				"Load Image1",
  				QDir::currentPath(),
  				"Image files (*.bmp *.png *.jpg *.gif);All files(*.*)");
  			image1=cv::imread(ImagePathName.toLatin1().data());
			if(image1.rows==0)
 				return;
 			parameters->fname0=(const char *)ImagePathName.toLocal8Bit();
			parameters[0].mask1=Mat::zeros(image1.rows,image1.cols,CV_8UC3);

  			ImagePathName = QFileDialog::getOpenFileName(
  					this,
  					"Load Image2",
  					QDir::currentPath(),
  					"Image files (*.bmp *.png *.jpg *.gif);All files(*.*)");
  			image2=cv::imread(ImagePathName.toLatin1().data());
 			if(image2.rows==0)
 				return;
			parameters->fname1=(const char *)ImagePathName.toLocal8Bit();
  			parameters[0].mask2=Mat::zeros(image2.rows,image2.cols,CV_8UC3);

			QString item;
			item.sprintf("&layer %d",0);
			show_layer[0]=new QAction(item.toLatin1().data(), this);
			layer_view->addAction(show_layer[0]);
			show_layer[0]->setCheckable(true);
			connect(show_layer[0],SIGNAL(triggered()),this,SLOT(Layer0()));

			layer_num++;
		}

		imageEditorL->setImage(image1);
		imageEditorR->setImage(image2);
		imageEditorM->setImage(imageEditorL->_image,imageEditorR->_image);
		ctrbar->_status=1;
		mod=1;

		for(int i=0;i<layer_num;i++)
		{
			int n=log((float)MIN(image1.cols,image1.rows))/log(2.0f)-log((float)parameters[i].start_res)/log(2.0f)+1;
  			pyramids[i].build_pyramid(image1,image2,parameters[i],n,i,gpu_cap);
		}
		thread_start(layer_index);

		Changelayer();
 	}
 }


 void MdiEditor::SaveProject()
 {

  	WriteXmlFile(pro_path+"\\settings.xml");

	
 }

 void MdiEditor::SaveVector()
 {
	 QString filename;
	 filename.sprintf("%s\\data.txt", pro_path.toLatin1().data());
	 QFile file(filename.toLatin1().data());
	 if (file.open(QIODevice::WriteOnly))
	 {
		 QTextStream out(&file);

		 for (int y = 0; y<image1.rows; y++)
			 for (int x = 0; x<image1.cols; x++)
			 {
			 Vec3f v = pyramids[layer_index]._vector.at<Vec3f>(y, x);
			 out << v[0] << " " << v[1] << endl;
			 }

		 file.flush();
		 file.close();
	 }
 }


 bool MdiEditor::ReadXmlFile(QString filename)
 {
  	QDomDocument doc("settings");
  	QFile file(filename);

  	if(file.open(QIODevice::ReadOnly))
  	{
  		doc.setContent(&file);
  		QDomElement root = doc.documentElement();
  		QDomElement child1=root.firstChildElement();

   		while(!child1.isNull())
   		{
  			if (child1.tagName()=="images")
  			{
  				QString ImagePathName = pro_path+child1.attribute("image1");
  				image1=cv::imread(ImagePathName.toLatin1().data());
				parameters->fname0=(const char *)ImagePathName.toLocal8Bit();

  				ImagePathName = pro_path+child1.attribute("image2");
  				image2=cv::imread(ImagePathName.toLatin1().data());
				parameters->fname1=(const char *)ImagePathName.toLocal8Bit();
  			}
 			else if (child1.tagName()=="layers")
 			{
 				layer_num=child1.attribute("num").toInt();
 				layer_index=0;
 				QDomElement child2=child1.firstChildElement();

				for(int i=0;i<layer_num;i++)
					parameters[i].fname0=parameters[0].fname0,parameters[i].fname1=parameters[0].fname1;
 				while (!child2.isNull())
 				{
 					int index=child2.tagName().remove("l").toInt();
 					QString item;
					item.sprintf("&layer %d",index);
					show_layer[index]=new QAction(item.toLatin1().data(), this);
					layer_view->addAction(show_layer[index]);
					show_layer[index]->setCheckable(true);
					switch(index)
					{
					case 0:
						connect(show_layer[index],SIGNAL(triggered()),this,SLOT(Layer0()));
						break;
					case 1:
						connect(show_layer[index],SIGNAL(triggered()),this,SLOT(Layer1()));
						break;
					case 2:
						connect(show_layer[index],SIGNAL(triggered()),this,SLOT(Layer2()));
						break;
					case 3:
						connect(show_layer[index],SIGNAL(triggered()),this,SLOT(Layer3()));
						break;
					case 4:
						connect(show_layer[index],SIGNAL(triggered()),this,SLOT(Layer4()));
						break;
					}

 					QDomElement child3=child2.firstChildElement();
 					while(!child3.isNull())
 					{
 						if(child3.tagName()=="masks")
 						{
 							QString ImagePathName = pro_path+child3.attribute("mask1");
 							parameters[index].mask1=cv::imread(ImagePathName.toLatin1().data());

 							ImagePathName = pro_path+child3.attribute("mask2");
 							parameters[index].mask2=cv::imread(ImagePathName.toLatin1().data());
 						}
 						else if (child3.tagName()=="parameters")
 						{
 							QDomElement elem=child3.firstChildElement();
 							while(!elem.isNull())
 							{
 								if(elem.tagName()=="weight")
 								{
 									parameters[index].w_ssim=elem.attribute("ssim").toFloat();
 									parameters[index].w_tps=elem.attribute("tps").toFloat();
 									parameters[index].w_ui=elem.attribute("ui").toFloat();
 									parameters[index].ssim_clamp=1.0-elem.attribute("ssimclamp").toFloat();
 								}
 								else if(elem.tagName()=="points")
 								{
									QString points=elem.attribute("image1");
									QStringList list1=points.split(" ");
									points=elem.attribute("image2");
									QStringList list2=points.split(" ");
									for (int i=0;i<list1.count()-1;i+=2)
									{
										ConstraintPoint elem;
										elem.lp.x=list1[i].toFloat();
										elem.lp.y=list1[i+1].toFloat();
										elem.rp.x=list2[i].toFloat();
										elem.rp.y=list2[i+1].toFloat();
										parameters[index].ui_points.push_back(elem);
									}
								}
 								else if(elem.tagName()=="boundary")
								{
									int cond=elem.attribute("lock").toInt();
									switch(cond)
									{
									case 0:
										parameters[index].bcond=BCOND_NONE;
										break;
									case 1:
										parameters[index].bcond=BCOND_CORNER;
										break;
									case 2:
										parameters[index].bcond=BCOND_BORDER;
										break;
									}
								}


 								else if(elem.tagName()=="debug")
 								{
 									parameters[index].max_iter=elem.attribute("iternum").toInt();
 									parameters[index].max_iter_drop_factor=elem.attribute("dropfactor").toFloat();
 									parameters[index].eps=elem.attribute("eps").toFloat();
 									parameters[index].start_res=elem.attribute("startres").toInt();
 								}

 								elem=elem.nextSiblingElement();
 							}
 					}
 					child3=child3.nextSiblingElement();
 				}

  				child2=child2.nextSiblingElement();
  			}

  		}

   		child1=child1.nextSiblingElement();
 	}
  		file.close();
  		return true;
  }
 	return false;
 }

 bool MdiEditor::WriteXmlFile(QString filename)
 {
 	//getParameters();
  	QFile file(filename);
  	if(file.open(QIODevice::WriteOnly | QIODevice::Truncate |QIODevice::Text))
  	{
  		QString str;
  		QDomDocument doc;
  		QDomText text;
  		QDomElement element;
  		QDomAttr attribute;

  		QDomProcessingInstruction instruction = doc.createProcessingInstruction("xml","version=\'1.0\'");
  		doc.appendChild(instruction);

   		QDomElement root = doc.createElement("project");
   		doc.appendChild(root);

  		//Images
   		QDomElement eimages=doc.createElement("images");
   		root.appendChild(eimages);

   		attribute=doc.createAttribute("image1");
  		attribute.setValue("/image1.png");

  		eimages.setAttributeNode(attribute);
		QString filename;
		filename.sprintf("%s/image1.png",pro_path.toLatin1().data());
		cv::imwrite(filename.toLatin1().data(),image1);

  		attribute=doc.createAttribute("image2");

  		attribute.setValue("\\image2.png");


  		eimages.setAttributeNode(attribute);

		filename.sprintf("%s\\image2.png",pro_path.toLatin1().data());

		cv::imwrite(filename.toLatin1().data(),image2);

   		//layer
 		QDomElement elayers=doc.createElement("layers");
 		root.appendChild(elayers);

 		attribute=doc.createAttribute("num");
 		attribute.setValue(str.sprintf("%d",layer_num));
 		elayers.setAttributeNode(attribute);

 		for(int index=0;index<layer_num;index++)
 		{
 			QDomElement elayer=doc.createElement(str.sprintf("l%d",index));
 			elayers.appendChild(elayer);

			QDir dir;
			QString filename;

			filename.sprintf("%s\\%d",pro_path.toLatin1().data(),index);

			dir.mkdir(filename);

			//masks
 			QDomElement emask=doc.createElement("masks");
 			elayer.appendChild(emask);

 			attribute=doc.createAttribute("mask1");

 			attribute.setValue(str.sprintf("\\%d\\mask1.png",index));

 			emask.setAttributeNode(attribute);

			filename.sprintf("%s\\%d\\mask1.png",pro_path.toLatin1().data(),index);

			cv::imwrite(filename.toLatin1().data(),parameters[index].mask1);

 			attribute=doc.createAttribute("mask2");

 			attribute.setValue(str.sprintf("\\%d\\mask2.png",index));

 			emask.setAttributeNode(attribute);

			filename.sprintf("%s\\%d\\mask2.png",pro_path.toLatin1().data(),index);

			cv::imwrite(filename.toLatin1().data(),parameters[index].mask2);


 			//parameters
 			QDomElement epara=doc.createElement("parameters");
 			elayer.appendChild(epara);

 			//weight
 			element=doc.createElement("weight");
 			epara.appendChild(element);

 			attribute=doc.createAttribute("ssim");
 			attribute.setValue(str.sprintf("%f",parameters[index].w_ssim));
 			element.setAttributeNode(attribute);

 			attribute=doc.createAttribute("tps");
 			attribute.setValue(str.sprintf("%f",parameters[index].w_tps));
 			element.setAttributeNode(attribute);

 			attribute=doc.createAttribute("ui");
 			attribute.setValue(str.sprintf("%f",parameters[index].w_ui));
 			element.setAttributeNode(attribute);

 			attribute=doc.createAttribute("ssimclamp");
 			attribute.setValue(str.sprintf("%f",1.0-parameters[index].ssim_clamp));
 			element.setAttributeNode(attribute);

 			//control points
 			element=doc.createElement("points");
 			epara.appendChild(element);

 			attribute=doc.createAttribute("image1");
 			str="";
 			for(size_t i=0;i<parameters[index].ui_points.size();i++)
 			{
 				QString num;
 				str.append(num.sprintf("%f ",parameters[index].ui_points[i].lp.x));
 				str.append(num.sprintf("%f ",parameters[index].ui_points[i].lp.y));
 			}
 			attribute.setValue(str);
 			element.setAttributeNode(attribute);


 			attribute=doc.createAttribute("image2");
 			str="";
 			for(size_t i=0;i<parameters[index].ui_points.size();i++)
 			{
 				QString num;
 				str.append(num.sprintf("%f ",parameters[index].ui_points[i].rp.x));
 				str.append(num.sprintf("%f ",parameters[index].ui_points[i].rp.y));
 			}
 			attribute.setValue(str);
 			element.setAttributeNode(attribute);

 			//boundary
 			element=doc.createElement("boundary");
 			epara.appendChild(element);

 			attribute=doc.createAttribute("lock");
			int bcond=0;
			switch(parameters[index].bcond)
			{
			case BCOND_NONE:
				bcond=0;
				break;
			case BCOND_CORNER:
				bcond=1;
				break;
			case BCOND_BORDER:
				bcond=2;
				break;
			}
 			attribute.setValue(str.sprintf("%d",bcond));
 			element.setAttributeNode(attribute);


 			//debug
 			element=doc.createElement("debug");
 			epara.appendChild(element);

 			attribute=doc.createAttribute("iternum");
 			attribute.setValue(str.sprintf("%d",parameters[index].max_iter));
 			element.setAttributeNode(attribute);

 			attribute=doc.createAttribute("dropfactor");
 			attribute.setValue(str.sprintf("%f",parameters[index].max_iter_drop_factor));
 			element.setAttributeNode(attribute);

 			attribute=doc.createAttribute("eps");
 			attribute.setValue(str.sprintf("%f",parameters[index].eps));
 			element.setAttributeNode(attribute);

 			attribute=doc.createAttribute("startres");
 			attribute.setValue(str.sprintf("%d",parameters[index].start_res));
 			element.setAttributeNode(attribute);

 		}

   		QTextStream out(&file);
  		out.setCodec("UTF-8");
  		doc.save(out,4);

  		file.close();
 		return true;
  	}

 	return false;
 }

 void MdiEditor::Changelayer()
 {
	 imageEditorL->clear();
	 imageEditorR->clear();
	 imageEditorL->setParameters(&parameters[layer_index],parameters[layer_index].mask1);
	 imageEditorR->setParameters(&parameters[layer_index],parameters[layer_index].mask2);
	 imageEditorM->setImage(imageEditorL->_image_layer,imageEditorR->_image_layer);
	 imageEditorM->setParameters(&parameters[layer_index]);
	 SetResults(layer_index);

 }
 void MdiEditor::SetResults(int index)
 {
	 if(layer_index==index)
	 {
		 imageEditorM->setImage(pyramids,index,index);
		 QString path;

		 path.sprintf("%s\\%d",pro_path.toLatin1().data(),index);

		 imageEditorA->set(path,pyramids,index,index,&parameters[index]);

		 if (thread_flag[index]==0)
		 {
			 delete readyLabel;
			 readyLabel = new QLabel(tr("Please create/load a project"));
			 statusBar()->addWidget(readyLabel, 1);
			 imageDockEditorM->setPalette(QPalette(QColor ( 219, 219, 219 )));
			 imageDockEditorA->setPalette(QPalette(QColor ( 219, 219, 219 )));
		 }
		 else if(thread_flag[index]==1)
		 {
			 float percentage = 0;
			 if(gpu_flag&&match_thread_GPU[layer_index])
				percentage=(float)match_thread_GPU[index]->_current_iter/(float)match_thread_GPU[index]->_total_iter*100;
			if(!gpu_flag&&match_thread[layer_index])
				percentage=(float)match_thread[index]->_current_iter/(float)match_thread[index]->_total_iter*100;
			 delete readyLabel;
			 QString str;
			 str.sprintf("optimizing %f%%",percentage);
			 readyLabel = new QLabel(str);
			 statusBar()->addWidget(readyLabel);
			 imageDockEditorM->setPalette(QPalette(QColor ( 255, 0, 0 )));
			 imageDockEditorA->setPalette(QPalette(QColor ( 255, 0, 0 )));
		 }

		 else if(thread_flag[index]<4)
		 {
			 delete readyLabel;
			 readyLabel = new QLabel(tr("optimization finished, post-processing"));
			 statusBar()->addWidget(readyLabel, 1);
			 imageDockEditorM->setPalette(QPalette(QColor ( 255, 255, 0 )));
			 imageDockEditorA->setPalette(QPalette(QColor ( 255, 255, 0 )));
		 }
		 else
		 {
			 delete readyLabel;
			 readyLabel = new QLabel(tr("Completed"));
			 statusBar()->addWidget(readyLabel, 1);
			 imageDockEditorM->setPalette(QPalette(QColor ( 0, 255, 0 )));
			 imageDockEditorA->setPalette(QPalette(QColor ( 0, 255, 0 )));

			 if (_auto)
			 {
				 ctrbar->_status=0;
				 imageEditorA->StatusChange(0);
			 }
		 }

		  updateALL();
	 }
 }

 void MdiEditor::ModifyPara()
 {
	 CDlgPara dlg(parameters[layer_index]);
	 connect(&dlg,SIGNAL(sigModified()),this,SLOT(PtModified()));
	 dlg.exec();

 }
 void MdiEditor::PtModified()
 {
	 pyramids[layer_index]._gpu->params()=parameters[layer_index];
	 thread_start(layer_index);
 }

  void MdiEditor::thread_start(int index)
  {
	 DeleteThread(index);

	if(index>=0&&index<layer_num)
	{
		thread_flag[index]=0;
		if(gpu_flag)
		{
			match_thread_GPU[index] = new CMatchingThread_GPU(parameters[index],pyramids[index],index);
			connect(match_thread_GPU[index],SIGNAL(sigFinished(int)),this,SLOT(thread_finished(int)));
			connect(match_thread_GPU[index],SIGNAL(sigUpdate(int)),this,SLOT(SetResults(int)));
			match_thread_GPU[index]->start(QThread::HighestPriority);
		}
		else
		{
			match_thread[index] = new CMatchingThread(parameters[index],pyramids[index],index);
			connect(match_thread[index],SIGNAL(sigFinished(int)),this,SLOT(thread_finished(int)));
			connect(match_thread[index],SIGNAL(sigUpdate(int)),this,SLOT(SetResults(int)));
			match_thread[index]->start(QThread::HighestPriority);
		}

		thread_flag[index]=1;
	}

  }

 void MdiEditor::thread_finished(int index)
 {

	thread_flag[index]=2;
	if(index==layer_index)
		SetResults(index);

 	if(index==0)
 	{
		cv::Mat extends1=cv::Mat(image1.rows,image1.cols,CV_8UC4);
		cv::Mat extends2=cv::Mat(image2.rows,image2.cols,CV_8UC4);
		int from_to[] = { 0,0,1,1,2,2,4,3 };
		cv::Mat src1[2]={image1,parameters[index].mask1};
		cv::Mat src2[2]={image2,parameters[index].mask2};
		cv::mixChannels(src1, 2, &extends1, 1, from_to, 4 );
		cv::mixChannels(src2, 2, &extends2, 1, from_to, 4 );
		int ex=image1.rows*0.2;
		pyramids[index]._extends1=cv::Mat(image1.rows+ex*2,image1.cols+ex*2,CV_8UC4,Scalar(255,255,255,255));
		pyramids[index]._extends2=cv::Mat(image2.rows+ex*2,image2.cols+ex*2,CV_8UC4,Scalar(255,255,255,255));
		extends1.copyTo(pyramids[index]._extends1(Rect(ex, ex, image1.cols, image1.rows)));
		extends2.copyTo(pyramids[index]._extends2(Rect(ex, ex, image2.cols, image2.rows)));

		poison_thread[index]=new CPoissonExt(index,pyramids[index]._vector,pyramids[index]._extends1,pyramids[index]._extends2,gpu_flag);
		connect(poison_thread[index],SIGNAL(sigFinished(int)),this,SLOT(poisson_finished(int)));
		poison_thread[index]->start(QThread::HighestPriority);
 	}
 	else
 		poisson_finished(index);


 	qpath_thread[index]=new CQuadraticPath(index,pyramids[index]._vector,pyramids[index]._qpath,gpu_flag);
 	connect(qpath_thread[index],SIGNAL(sigFinished(int)),this,SLOT(qpath_finished(int)));
 	qpath_thread[index]->start(QThread::HighestPriority);

	//poisson_finished(index);
	//qpath_finished(index);
	//AutoQuit();

 }

void MdiEditor::poisson_finished(int index)
{
	thread_flag[index]++;
	if(index==layer_index)
		SetResults(index);
}
 void MdiEditor::qpath_finished(int index)
 {
	 thread_flag[index]++;
	 if(index==layer_index)
		 SetResults(index);
 }


 void MdiEditor::ModifyMask()
 {

	 imageEditorL->_flag_mask=imageEditorR->_flag_mask=true;
	 imageEditorL->_scissor=imageEditorR->_scissor=true;
	 mod=2;
	 updateALL();
 }



 void MdiEditor::ShowMask()
 {
	 imageEditorL->_flag_mask=imageEditorR->_flag_mask=true;
	 show_mask->setChecked(true);
	 hide_mask->setChecked(false);
	 updateALL();
 }
 void MdiEditor::HideMask()
 {
	 imageEditorL->_flag_mask=imageEditorR->_flag_mask=false;

	 updateALL();
 }
 void MdiEditor::ShowHalfway()
 {
	 imageEditorM->_flag_error=false;

	 updateALL();
 }
 void MdiEditor::ShowError()
 {
	 imageEditorM->_flag_error=true;

	 updateALL();
 }
 void MdiEditor::ColorFromImage1()
 {
	 imageEditorA->_colorfrom=0;

	 updateALL();

 }
 void MdiEditor::ColorFromImage12()
 {
	 imageEditorA->_colorfrom=1;

	 updateALL();
 }
 void MdiEditor::ColorFromImage2()
 {
	 imageEditorA->_colorfrom=2;
	 updateALL();
 }

 void MdiEditor::contextMenuEvent(QContextMenuEvent* e)
 {
 	 if(mod==2||mod==3)
 	 {
		 QMenu *menu = new QMenu();
		 menu->addAction(cancel);
		 menu->addSeparator();
		 menu->addAction(confirm);
		 menu->addSeparator();
		 menu->exec(e->globalPos());
		 delete menu;
	 }
 }

 void MdiEditor::Confirm()
 {
	 DeleteThread(layer_index);
	 switch(mod)
	 {
	 case 2:

		 if(imageEditorL->_contourList.size()>0||imageEditorR->_contourList.size()>0)
		 {
			 Point polygonlist[5][200];
			 int polygonnum[5];
			 for (int i=0;i<imageEditorL->_contourList.size();i++)
			 {
				 polygonnum[i]=0;
				 for(int j=0;j<imageEditorL->_contourList[i].size();j+=3)
				 {
					 int x=imageEditorL->_contourList[i][j].x()*image1.cols;
					 int y=imageEditorL->_contourList[i][j].y()*image1.rows;
					 polygonlist[i][polygonnum[i]]=Point(x,y);
					 polygonnum[i]++;
				 }
			 }

			 const Point* polygonlistL[5] = { polygonlist[0],polygonlist[1],polygonlist[2],polygonlist[3],polygonlist[4]};
			 int polygonnumL[] = {polygonnum[0],polygonnum[1],polygonnum[2],polygonnum[3],polygonnum[4]};
			 if(imageEditorL->_contourList.size()>0)
			 cv::fillPoly(parameters[layer_index].mask1,polygonlistL,polygonnumL,imageEditorL->_contourList.size(),Scalar(255,0,0));



			 for (int i=0;i<imageEditorR->_contourList.size();i++)
			 {
				 polygonnum[i]=0;
				 for(int j=0;j<imageEditorR->_contourList[i].size();j+=3)
				 {
					 int x=imageEditorR->_contourList[i][j].x()*image1.cols;
					 int y=imageEditorR->_contourList[i][j].y()*image1.rows;
					 polygonlist[i][polygonnum[i]]=Point(x,y);
					 polygonnum[i]++;
				 }
			 }

			 const Point* polygonlistR[5] = { polygonlist[0],polygonlist[1],polygonlist[2],polygonlist[3],polygonlist[4]};
			 int polygonnumR[] = {polygonnum[0],polygonnum[1],polygonnum[2],polygonnum[3],polygonnum[4]};
			  if(imageEditorR->_contourList.size()>0)
			 cv::fillPoly(parameters[layer_index].mask2,polygonlistR,polygonnumR,imageEditorR->_contourList.size(),Scalar(255,0,0));

			 pyramids[layer_index].build_pyramid(parameters[layer_index].mask1,parameters[layer_index].mask2);
			 thread_start(layer_index);
		 }
		 break;

	 case 3:
		 parameters[layer_num] = parameters[layer_index];
		 if(imageEditorL->_contourList.size()>0||imageEditorR->_contourList.size()>0)
		 {
			 Point polygonlist[5][200];
			 int polygonnum[5];
			 for (int i=0;i<imageEditorL->_contourList.size();i++)
			 {
				 polygonnum[i]=0;
				 for(int j=0;j<imageEditorL->_contourList[i].size();j+=3)
				 {
					 int x=imageEditorL->_contourList[i][j].x()*image1.cols;
					 int y=imageEditorL->_contourList[i][j].y()*image1.rows;
					 polygonlist[i][polygonnum[i]]=Point(x,y);
					 polygonnum[i]++;
				 }
			 }

			 const Point* polygonlistL[5] = { polygonlist[0],polygonlist[1],polygonlist[2],polygonlist[3],polygonlist[4]};
			 int polygonnumL[] = {polygonnum[0],polygonnum[1],polygonnum[2],polygonnum[3],polygonnum[4]};
			 if(imageEditorL->_contourList.size()>0)
			 cv::fillPoly(parameters[layer_index].mask1,polygonlistL,polygonnumL,imageEditorL->_contourList.size(),Scalar(255,255,0));

			 parameters[layer_num].mask1=cv::Mat(image1.rows,image1.cols,CV_8UC3,Scalar(0,255,0));
			 if(imageEditorL->_contourList.size()>0)
			 cv::fillPoly(parameters[layer_num].mask1,polygonlistL,polygonnumL,imageEditorL->_contourList.size(),Scalar(0,0,0));

			 for (int i=0;i<imageEditorR->_contourList.size();i++)
			 {
				 polygonnum[i]=0;
				 for(int j=0;j<imageEditorR->_contourList[i].size();j+=3)
				 {
					 int x=imageEditorR->_contourList[i][j].x()*image1.cols;
					 int y=imageEditorR->_contourList[i][j].y()*image1.rows;
					 polygonlist[i][polygonnum[i]]=Point(x,y);
					 polygonnum[i]++;
				 }
			 }

			 const Point* polygonlistR[5] = { polygonlist[0],polygonlist[1],polygonlist[2],polygonlist[3],polygonlist[4]};
			 int polygonnumR[] = {polygonnum[0],polygonnum[1],polygonnum[2],polygonnum[3],polygonnum[4]};
			  if(imageEditorR->_contourList.size()>0)
			 cv::fillPoly(parameters[layer_index].mask2,polygonlistR,polygonnumR,imageEditorR->_contourList.size(),Scalar(255,255,0));
			 parameters[layer_num].mask2=cv::Mat(image2.rows,image2.cols,CV_8UC3,Scalar(0,255,0));
			  if(imageEditorR->_contourList.size()>0)
			 cv::fillPoly(parameters[layer_num].mask2,polygonlistR,polygonnumR,imageEditorR->_contourList.size(),Scalar(0,0,0));

			 pyramids[layer_index].build_pyramid(parameters[layer_index].mask1,parameters[layer_index].mask2);
			 thread_start(layer_index);
			 int n=log((float)MIN(image1.cols,image1.rows))/log(2.0f)-log((float)parameters[layer_num].start_res)/log(2.0f)+1;
			 pyramids[layer_num].build_pyramid(image1, image2, parameters[layer_num], n, layer_num, gpu_cap);

			 QString item;
			 item.sprintf("&layer %d",layer_num);
			 show_layer[layer_num]=new QAction(item.toLatin1().data(), this);
			 layer_view->addAction(show_layer[layer_num]);
			 show_layer[layer_num]->setCheckable(true);
			 switch(layer_num)
			 {
			 case 0:
				 connect(show_layer[layer_num],SIGNAL(triggered()),this,SLOT(Layer0()));
				 break;
			 case 1:
				 connect(show_layer[layer_num],SIGNAL(triggered()),this,SLOT(Layer1()));
				 break;
			 case 2:
				 connect(show_layer[layer_num],SIGNAL(triggered()),this,SLOT(Layer2()));
				 break;
			 case 3:
				 connect(show_layer[layer_num],SIGNAL(triggered()),this,SLOT(Layer3()));
				 break;
			 case 4:
				 connect(show_layer[layer_num],SIGNAL(triggered()),this,SLOT(Layer4()));
				 break;
			 }
			 layer_num++;

		 }
		break;

	 case 4:
		 break;
	 }

	 mod=1;
	 imageEditorL->_scissor=imageEditorR->_scissor=imageEditorM->_scissor=false;
	Changelayer();
 }

 void MdiEditor::Cancel()
 {
	 mod=1;
	 imageEditorL->_scissor=imageEditorR->_scissor=imageEditorM->_scissor=false;

	 imageEditorL->clear();
	 imageEditorR->clear();
	 updateALL();
 }

 void MdiEditor::ModifyLayer()
 {
	 imageEditorL->_scissor=imageEditorR->_scissor=true;
	 mod=3;
	 updateALL();
 }

 void MdiEditor::ReorderLayer(char name, bool up,float x, float y)
 {
	 int xx=x*image1.cols;
	 int yy=y*image1.rows;

	 int swap[2] = {-1,-1};
	 int order = -1;
	 for(int i=0;i<layer_num;i++)
	 {
		 Vec3b color;
		 if (name=='l'||name=='L')
			color=parameters[i].mask1.at<Vec3b>(yy,xx);
		 else
			color=parameters[i].mask2.at<Vec3b>(yy,xx);

		 if(color[1]==0)
		 {
			 swap[0]=i;
			 if(up)
				 order=MIN(MAX(pyramids[i]._order+1,0),layer_num-1);
			 else
				 order=MIN(MAX(pyramids[i]._order-1,0),layer_num-1);
			 break;
		 }
	 }

	 for(int i=0;i<layer_num;i++)
	 {
		 if(pyramids[i]._order==order)
		 {
			 swap[1]=i;
			 break;
		 }
	 }

         assert(swap[0] >= 0);
         assert(swap[1] >= 0);
         assert(order >= 0);

	 pyramids[swap[1]]._order=pyramids[swap[0]]._order;
	 pyramids[swap[0]]._order=order;
	 MultyLayer();
 }

 void MdiEditor::Layer0()
 {
	 DeleteThread(layer_index);
	 layer_index=0;
	 mod=1;
	 if(thread_flag[layer_index]<4)
		 thread_start(layer_index);
	 imageEditorL->_flag_multilayer=imageEditorR->_flag_multilayer=imageEditorM->_flag_multilayer=false;
	 Changelayer();

 }
 void MdiEditor::Layer1()
 {
	 DeleteThread(layer_index);
	 layer_index=1;
	 mod=1;
	 if(thread_flag[layer_index]<4)
 		 thread_start(layer_index);
	 imageEditorL->_flag_multilayer=imageEditorR->_flag_multilayer=imageEditorM->_flag_multilayer=false;
	 Changelayer();

 }
 void MdiEditor::Layer2()
 {
	 DeleteThread(layer_index);
	 layer_index=2;
	 mod=1;
	 if(thread_flag[layer_index]<4)
		 thread_start(layer_index);
	 imageEditorL->_flag_multilayer=imageEditorR->_flag_multilayer=imageEditorM->_flag_multilayer=false;
	 Changelayer();
 }
 void MdiEditor::Layer3()
 {
	 DeleteThread(layer_index);
	 layer_index=3;
	 mod=1;
	 if(thread_flag[layer_index]<4)
		 thread_start(layer_index);
	 imageEditorL->_flag_multilayer=imageEditorR->_flag_multilayer=imageEditorM->_flag_multilayer=false;
	 Changelayer();

 }
 void MdiEditor::Layer4()
 {
	 DeleteThread(layer_index);
	 layer_index=4;
	 mod=1;
	 if(thread_flag[layer_index]<4)
		 thread_start(layer_index);
	 imageEditorL->_flag_multilayer=imageEditorR->_flag_multilayer=imageEditorM->_flag_multilayer=false;
	 Changelayer();

 }

 void MdiEditor::MultyLayer()
 {
	 layer_index=-1;
	 mod=4;
	 imageEditorL->setMultiLayer(pyramids,layer_num);
	 imageEditorR->setMultiLayer(pyramids,layer_num);
	 imageEditorM->setImage(pyramids,0,layer_num-1);
	 imageEditorA->set(pro_path,pyramids,0,layer_num-1,NULL);
	 imageEditorL->_flag_mask=imageEditorR->_flag_mask=false;
	 imageEditorM->_flag_error=false;
	 imageEditorL->_flag_multilayer=imageEditorR->_flag_multilayer=imageEditorM->_flag_multilayer=true;
	 updateALL();
 }

 void MdiEditor::CPU_Alg()
 {
	 if (gpu_flag)
	 {
		for(int i=0;i<layer_num;i++)
		{
			DeleteThread(i);
			thread_flag[i]=0;
		}
		gpu_flag=false;
		thread_start(layer_index);
		update();

	 }
 }

 void MdiEditor::GPU_Alg()
 {
	 if (!gpu_flag)
	 {
	
			 for (int i = 0; i<layer_num; i++)
			 {
				 DeleteThread(i);
				 thread_flag[i] = 0;
			 }
			 gpu_flag = true;

			thread_start(layer_index);
			 update();	 

	 }
 }


 void MdiEditor::AutoQuit()
 {
	 if(_auto)
	 {

		 SaveProject();
		 QString filename;
		
		filename.sprintf("%s\\time.txt",pro_path.toLatin1().data());

		QFile file(filename.toLatin1().data());

		if (file.open(QFile::WriteOnly | QFile::Truncate))
		{
			QTextStream out(&file);
			QString line;
			if (match_thread_GPU[layer_index])
				line.sprintf("Optimizing time: %f ms \n",match_thread_GPU[layer_index]->run_time);
			if (match_thread[layer_index])
				line.sprintf("Optimizing time: %f ms \n",match_thread[layer_index]->run_time);
			out<<line;
			if (poison_thread[layer_index])
			{
				line.sprintf("Poisson time: %f ms \n", poison_thread[layer_index]->_runtime);
				out << line;
			}
			if (qpath_thread[layer_index])
			{
				line.sprintf("Quadratic path time: %f ms \n", qpath_thread[layer_index]->_runtime);
				out << line;
			}
			if (imageEditorA)
			{
				line.sprintf("Rendering time: %f ms \n", imageEditorA->_runtime);
				out << line;
			}
			

			if (match_thread[layer_index]){

				out << "\n";

				line.sprintf("level \t resolution \t iter \t time \n");
				out << line;

				for (int i = 0; i < match_thread[layer_index]->_total_l-1; i++)
				{
					line.sprintf("%d \t %dx%d \t %d \t %f \n", i, match_thread[layer_index]->_pyramids.levels[i].w, match_thread[layer_index]->_pyramids.levels[i].h, match_thread[layer_index]->iter_num[i], match_thread[layer_index]->layer_time[i]);
					out << line;
				}
			}
			

		}
		
		file.flush();
		file.close();
		clear();
		_app->quit();
	 }
 }

