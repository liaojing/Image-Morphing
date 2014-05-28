#pragma once
#include "ui_DlgPara.h"
#include "../Header.h"
class CDlgPara : public QDialog, private Ui::CDlgPara
{
	Q_OBJECT

public:
	CDlgPara(Parameters &para);
	~CDlgPara(void);

public:
signals:
	void sigModified();

public slots:
	void onConfirm();
	void onCancel();

protected:
	Parameters& _para;



};

