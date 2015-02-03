INCS_linux = -Iinclude -fopenmp
LIBS_linux = -L/usr/lib lib.linux/libpng.a -lz -lgomp -lm 

INCS_macosx = -Iinclude -fopenmp
LIBS_macosx = lib.macosx/libpng.a -lz -lgomp -lm 

PLAT?=linux

INCS=$(INCS_$(PLAT))
LIBS=$(LIBS_$(PLAT))

CXXFLAGS = -W -Wall -ffast-math -ansi -pedantic -O3 $(INCS) 
CFLAGS = -W -Wall -ffast-math -ansi -pedantic -O3 $(INCS)
CC=gcc
CXX=g++

LIBOBJS=error.o discrete.o util.o dlti.o chronos.o scale.o image.o

OBJS = pyramid.o libpyramid.a

all: libpyramid.a

libpyramid.a: $(LIBOBJS)
	ar cru $@ $^

pyramid: $(OBJS)
	$(CXX) -o $@ $^ $(LIBS)

run: pyramid
	./pyramid cactus-2048.png

clean:
	rm -f pyramid $(OBJS) 
