EIGEN_INCLUDE=-I /opt/local/include/eigen3
CIMG_INCLUDE=-I /opt/local/include
LIBX11=-L/opt/local/lib -lX11
INCLUDE=$(EIGEN_INCLUDE) $(CIMG_INCLUDE) -I include
CC=c++
CFLAGS=-g -O2

LEARCH_SOURCES=LEARCHCompileTest.cc grid_2d/LEARCH2DGrid.cc
SOURCES=$(LEARCH_SOURCES)
OBJECTS=$(SOURCES:.cc=.o)

LIBS=

all: LEARCH2DGrid

%.o: %.cc
	$(CC) $(CFLAGS) $(INCLUDE) -c $< -o $@

%.d: %.cc
	@set -e; rm -f $@; \
	$(CC) -M -MT $(@:.d=.o) $(INCLUDE) $< > $@.$$$$; \
	sed 's,\($*\)\.o[ :]*,\1.o $@ : ,g' < $@.$$$$ > $@; \
	rm -f $@.$$$$

include $(SOURCES:.cc=.d)

clean:
	rm $(OBJECTS) $(SOURCES:.cc=.d)

LEARCHCompileTest: LEARCHCompileTest.o $(LEARCH_OBJECTS)
	c++ $(INCLUDE) $^ -o $@

LEARCH2DGrid: grid_2d/LEARCH2DGrid.o grid_2d/fastMarching.o
	$(CC) $(CFLAGS) $(INCLUDE) $^ -o $@ $(LIBX11) -lpthread
