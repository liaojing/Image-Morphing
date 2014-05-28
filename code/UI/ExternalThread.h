#pragma once

#include <QThread>


class CExternalThread : public QThread
{
protected:

	void run();

};

