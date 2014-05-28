#pragma once
#pragma once
#include "ui_CtrBar.h"
#include "../Header.h"
class CCtrBar: public QWidget, private Ui::CCtrBar
{
	Q_OBJECT
public:
	CCtrBar(void);
	~CCtrBar(void);
	void paintEvent(QPaintEvent *event);

public:
signals:
	void sigRangeChange(int range);
	void sigStatusChange(int status);

public slots:
	void range_changed(int value);
	void play();
	void record();
	void stop();
	void pause();
	void record_finished();
public:
	int _status; //0-record,1-play,2-pause,3-stop
	int _range;

};

