#include "DlgPara.h"


CDlgPara::CDlgPara(Parameters &para):_para(para)
{
	setupUi(this);
	
	QString str = QString("%1").arg(_para.w_ssim);
	lineEdit_SSIM->setText(str);
	str = QString("%1").arg(_para.w_tps);
	lineEdit_TPS->setText(str);
	str = QString("%1").arg(1.0-_para.ssim_clamp);
	lineEdit_SSIMClamp->setText(str);
	str = QString("%1").arg(_para.w_ui);
	lineEdit_UI->setText(str);

	str = QString("%1").arg(_para.max_iter);
	lineEdit_Iter->setText(str);
	str = QString("%1").arg(_para.max_iter_drop_factor);
	lineEdit_Drop->setText(str);
	str = QString("%1").arg(_para.eps);
	lineEdit_EPS->setText(str);
		
	switch (_para.bcond)
	{
	case BCOND_NONE:
		radioButton_1->setChecked(true);
		break;
	case BCOND_CORNER:
		radioButton_2->setChecked(true);
		break;
	case BCOND_BORDER:
		radioButton_3->setChecked(true);
		break;	
	}

	connect(pushButton_Confirm,SIGNAL(clicked()),this,SLOT(onConfirm()));
	connect(pushButton_Cancel,SIGNAL(clicked()),this,SLOT(onCancel()));
}


CDlgPara::~CDlgPara(void)
{
	
}

void CDlgPara::onConfirm()
{
	QString str = lineEdit_SSIM->text();
	_para.w_ssim=str.toFloat();
	str = lineEdit_TPS->text();
	_para.w_tps=str.toFloat();
	str = lineEdit_UI->text();
	_para.w_ui=str.toFloat();
	str = lineEdit_SSIMClamp->text();
	_para.ssim_clamp=1.0-str.toFloat();
	str = lineEdit_Iter->text();
	_para.max_iter=str.toFloat();
	str = lineEdit_Drop->text();
	_para.max_iter_drop_factor=str.toFloat();
	str = lineEdit_EPS->text();
	_para.eps=str.toFloat();
	
	if (radioButton_1->isChecked ())
		_para.bcond=BCOND_NONE;
	else if(radioButton_2->isChecked ())
		_para.bcond=BCOND_CORNER;
	else if(radioButton_3->isChecked())
		_para.bcond=BCOND_BORDER;	

	emit sigModified();
	close();
}

void CDlgPara::onCancel()
{
	
	close();
}
