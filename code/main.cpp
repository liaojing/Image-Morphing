#include "UI/MdiEditor.h"


int main(int argc, char *argv[])
{
	 QApplication app(argc, argv);

	MdiEditor mainWin(&app);
    if(argc>1)
	{
		mainWin._auto=true;
		mainWin.pro_path.sprintf("%s",argv[1]);
		mainWin.NewProject();
	}
	else
		mainWin._auto=false;

	mainWin.show();
    return app.exec();
}

