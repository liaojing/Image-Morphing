#include "UI/MdiEditor.h"


int main(int argc, char *argv[])
{
	
	QApplication app(argc, argv);
	QString work_path;
	work_path.sprintf("%s", argv[0]);
	int index = work_path.lastIndexOf("\\");
	if (index > 0)
	{
		work_path.remove(index, work_path.length() - index);
		QDir::setCurrent(work_path);
	}


	MdiEditor mainWin(&app);
	QString pro_path;
	pro_path.sprintf("%s", argv[argc-1]);
	if (pro_path.endsWith(".xml"))
	{
		int index=pro_path.lastIndexOf("\\");
		pro_path.remove(index, pro_path.length()-index);
		mainWin._auto=true;
		mainWin.pro_path=pro_path;
		mainWin.NewProject();
	}
	else
		mainWin._auto=false;
	

	mainWin.show();
    return app.exec();
}

