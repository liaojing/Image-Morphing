#include "ExternalThread.h"
#include <QProcess>

void CExternalThread::run()
{
	QProcess process;

	QStringList arguments;
	arguments<<"/k"<<"all.bat";
	process.startDetached("cmd",arguments);	
	
}
