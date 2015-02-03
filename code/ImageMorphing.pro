TEMPLATE      = app
HEADERS       = Header.h \
		UI/ImageEditor.h \
        	UI/MdiEditor.h \
		UI/HalfwayImage.h \
   		UI/ExternalThread.h \
		UI/RenderWidget.h \		
		UI/DlgPara.h \
		UI/CtrBar.h \
    		Algorithm/MatchingThread.h \  
		Algorithm/MatchingThread_GPU.h \  
		Algorithm/QuadraticPath.h \   
		Algorithm/Pyramid.h \    
		Algorithm/PoissonExt.h \       		
		IntelligentScissor/fibheap.h \
		IntelligentScissor/CostGraph.h \
		IntelligentScissor/CostNode.h
		
SOURCES       =  main.cpp \
		UI/MdiEditor.cpp \               
        	UI/ImageEditor.cpp \
		UI/HalfwayImage.cpp \
   		UI/ExternalThread.cpp \
		UI/RenderWidget.cpp \
		UI/DlgPara.cpp \
		UI/CtrBar.cpp \
		Algorithm/MatchingThread.cpp \
		Algorithm/MatchingThread_GPU.cpp \  	
		Algorithm/QuadraticPath.cpp \  
		Algorithm/Pyramid.cpp \   
		Algorithm/PoissonExt.cpp \     		
		IntelligentScissor/fibheap.cpp \
		IntelligentScissor/CostGraph.cpp \
		IntelligentScissor/CostNode.cpp

FORMS		  = UI/DlgPara.ui \
		    UI/CtrBar.ui

RESOURCES      =  UI/CtrBar.qrc
QT += xml
QT += opengl
CONFIG += 64bit