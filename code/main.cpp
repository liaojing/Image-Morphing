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

	QString para1;
	if (argc > 1) para1.sprintf("%s", argv[1]);
	QString para2;
	if (argc > 2) para2.sprintf("%s", argv[2]);

	mainWin._auto = false;
	if (argc > 2)
	{
		if (para1=="-auto" )
		{
			QString temp;
			temp = para1;
			para1 = para2;
			para2 = temp;
		}

		if (para2=="-auto")
			mainWin._auto = true;
	}

	
	if (argc > 1)
	{
		
		if (para1.endsWith(".xml"))
		{
			int index = para1.lastIndexOf("\\");
			para1.remove(index, para1.length() - index);
			mainWin.pro_path = para1;
			mainWin.NewProject(true);
		}
		else
			mainWin.pro_path.sprintf("%s", "");
	}
	
	mainWin.show();
    return app.exec();
}

