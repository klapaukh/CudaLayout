CUDADIR=
HOST:= $(shell uname -n)
ifeq ("$(HOST)","red-tomatoes")
	CUDADIR=/opt/cuda50
	NCFLAGS=-gencode arch=compute_10,code=sm_10
else
	CUDADIR=/opt/cuda
	NCFLAGS=-gencode arch=compute_20,code=sm_20 -gencode arch=compute_20,code=sm_21
endif

OBJECTS=main.o layout.co graph.o graphmlReader.o debug.o
TARGET=layout
CC=g++
NVCC=$(CUDADIR)/bin/nvcc
FLAGS=
CFLAGS=-pedantic -Wall -Wextra -lint -I$(CUDADIR)/include
LDFLAGS=-Xlinker -rpath $(CUDADIR)/lib64 -L$(CUDADIR)/lib64 -lcudart -lexpat -lGL -lGLU -lglut
NCFLAGS+=-m64 -I$(CUDADIR)/include 

release: CFLAGS += -O3
release: NCFLAGS += -O3
release: LDFLAGS += -O3
release: all

dbg: CFLAGS += -g -pg
dbg: LDFLAGS += -pg
dbg: NCFLAGS += -G -pg -Xcompiler -rdynamic -lineinfo
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
