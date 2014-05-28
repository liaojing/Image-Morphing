#include "CtrBar.h"


CCtrBar::CCtrBar(void)
{
	setAttribute(Qt::WA_StaticContents);
	setupUi(this);
	Slider_range->setRange(0,50);
	_status=-1;
	_range=50;
	Slider_range->setValue(_range);

	connect(Slider_range,SIGNAL(valueChanged(int)),this,SLOT(range_changed(int)));
	connect(pushButton_record,SIGNAL(clicked()),this,SLOT(record()));
	connect(pushButton_play,SIGNAL(clicked()),this,SLOT(play()));
	connect(pushButton_pause,SIGNAL(clicked()),this,SLOT(pause()));
	connect(pushButton_stop,SIGNAL(clicked()),this,SLOT(stop()));


}


CCtrBar::~CCtrBar(void)
{
}

void CCtrBar::paintEvent(QPaintEvent *event)
{
	switch(_status)
	{
	case -1:
		pushButton_record->setEnabled(false);
		pushButton_play->setEnabled(false);
		pushButton_pause->setEnabled(false);
		pushButton_stop->setEnabled(false);
		Slider_range->setEnabled(false);
		break;
	case 0:
		pushButton_record->setEnabled(false);
		pushButton_play->setEnabled(false);
		pushButton_pause->setEnabled(false);
		pushButton_stop->setEnabled(false);
		Slider_range->setEnabled(false);
		break;
	case 1:
		pushButton_record->setEnabled(true);
		pushButton_play->setEnabled(false);
		pushButton_pause->setEnabled(true);
		pushButton_stop->setEnabled(true);
		Slider_range->setEnabled(true);
		break;
	case 2:
		pushButton_record->setEnabled(true);
		pushButton_play->setEnabled(true);
		pushButton_pause->setEnabled(false);
		pushButton_stop->setEnabled(true);
		Slider_range->setEnabled(true);
		break;
	case 3:
		pushButton_record->setEnabled(true);
		pushButton_play->setEnabled(true);
		pushButton_pause->setEnabled(false);
		pushButton_stop->setEnabled(false);
		Slider_range->setEnabled(true);
		break;
	}
	
	
}

void CCtrBar::range_changed(int value)
{
	_range=value;
	emit sigRangeChange(_range);
	update();

}
void CCtrBar::play()
{
	_status=1;
	emit sigStatusChange(_status);
	update();
}
void CCtrBar::record()
{
	_status=0;
	emit sigStatusChange(_status);
	update();
}
void CCtrBar::stop()
{
	_status=3;
	emit sigStatusChange(_status);
	update();
}
void CCtrBar::pause()
{
	_status=2;
	emit sigStatusChange(_status);
	update();
}

void CCtrBar::record_finished()
{
	_status=1;
	update();
}