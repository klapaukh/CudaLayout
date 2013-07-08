CUDADIR=
HOST:= $(shell uname -n)
ifeq ("$(HOST)","red-tomatoes")
	CUDADIR=/opt/cuda50
else
	CUDADIR=/opt/cuda
endif

OBJECTS=main.o layout.co graph.o graphmlReader.o debug.o
TARGET=layout
CC=g++
NVCC=$(CUDADIR)/bin/nvcc
CFLAGS=-pedantic -Wall -Wextra -lint -I$(CUDADIR)/include -O2 #-g -pg
LDFLAGS=-Xlinker -rpath $(CUDADIR)/lib64 -L$(CUDADIR)/lib64 -lcudart -lexpat -lGL -lGLU -lglut #-pg
NCFLAGS=-m64 -I$(CUDADIR)/include -gencode arch=compute_20,code=sm_20 -gencode arch=compute_20,code=sm_21 -O2#-g -pg

all: $(TARGET)

$(TARGET): $(OBJECTS)
	g++ $(FLAGS) $(OBJECTS) -o $(TARGET) $(LDFLAGS)

%.o: %.cpp
	$(CC) -o $@ -c $(CFLAGS) $<

%.co: %.cu
	$(NVCC) -o $@ -c $(NCFLAGS) $<

clean:
	rm -rf $(OBJECTS) layout
