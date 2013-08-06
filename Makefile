CUDADIR=
HOST:= $(shell uname -n)
ifeq ("$(HOST)","red-tomatoes")
	CUDADIR=/opt/cuda50
	NCFLAGS=-gencode arch=compute_13,code=sm_13 -Xopencc "-W -Wall -Wextra -lint -pedantic"

else
	CUDADIR=/opt/cuda
	NCFLAGS=-gencode arch=compute_20,code=sm_20 -gencode arch=compute_20,code=sm_21
endif

OBJECTS=bitarray.o main.o layout.co graph.o graphmlReader.o debug.o
TARGET=layout
CC=g++
NVCC=$(CUDADIR)/bin/nvcc
FLAGS=
CFLAGS=-pedantic -Wall -Wextra -lint -I$(CUDADIR)/include
LDFLAGS=-Xlinker -rpath $(CUDADIR)/lib64 -L$(CUDADIR)/lib64 -lcudart -lexpat -lGL -lGLU -lglut
NCFLAGS+=-m64 -I$(CUDADIR)/include -Xcompiler "-Wall -Wextra -lint -DDEBUG"

release: CFLAGS += -O3
release: NCFLAGS += -O3 -use_fast_math
release: LDFLAGS += -O3
release: all

dbg: CFLAGS += -g -pg -DDEBUG
dbg: LDFLAGS += -pg
dbg: NCFLAGS += -g -G -pg --ptxas-options=-v -Xcompiler -rdynamic -lineinfo
dbg: all

all: $(TARGET)


$(TARGET): $(OBJECTS)
	g++ $(FLAGS) $(OBJECTS) -o $(TARGET) $(LDFLAGS)

%.o: %.cpp
	$(CC) -o $@ -c $(CFLAGS) $<

%.co: %.cu
	$(NVCC) -o $@ -c $(NCFLAGS) $<

clean:
	rm -rf $(OBJECTS) layout
