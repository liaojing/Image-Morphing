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
		IntelligentScissor/CostNode.h \
		gpumorph\src\imgio.h\
		gpumorph\src\pyramid.h\
		gpumorph\src\morph.h\
		gpumorph\src\stencil.h\
		CUDA/cg.cuh
	
		
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
		gpumorph/src/cpuoptim.cpp \
		gpumorph/src/downsample.cpp \
		gpumorph/src/imgio.cpp \
		gpumorph/src/morph.cu\
		gpumorph/src/render.cu\
		gpumorph/src/stencil.cpp \
		gpumorph/src/upsample.cu \
		CUDA/cg.cu

FORMS		  = UI/DlgPara.ui \
		    UI/CtrBar.ui

RESOURCES      =  UI/CtrBar.qrc
QT += xml
QT += opengl
CONFIG += 64bit
unix:INCLUDEPATH += /usr/local/cuda/include
unix:INCLUDEPATH += gpumorph/include
unix:INCLUDEPATH += /opt/intel/mkl/include
unix:INCLUDEPATH += pyramids/include
unix:LIBS += -lopencv_core -lopencv_highgui -lopencv_imgproc -fopenmp -lGLEW -lGLU -L/opt/intel/lib/intel64 -liomp5 -L/opt/intel/composerxe/mkl/lib/intel64 -lmkl_core -lmkl_blacs_intelmpi_lp64 -lmkl_scalapack_lp64 -lmkl_def -lmkl_intel_thread -lmkl_intel_lp64 -lmpi -llapack -lglut -L/usr/local/cuda/lib64 -lcudart ../pyramids/libpyramid.a -L../gpumorph/src -lmorph -L../gpumorph/lib/util -lutil -LCUDA -lcg -L../gpumorph/lib/resample -lresample
unix:QMAKE_CXXFLAGS += -fopenmp

CONFIG(32bit) {
    TARGET = 32bit_binary
    QMAKE_CXXFLAGS += -m32
    LIBS += -L<path to 32bit libraries>
}
CONFIG(64bit) {
    TARGET = 64bit_binary
}
