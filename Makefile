OBJECTS=main.o layout.co graph.o graphmlReader.o debug.o
TARGET=layout
CC=g++
NVCC=/opt/cuda/bin/nvcc
CFLAGS=-pedantic -Wall -Wextra -lint -I/opt/cuda/include -g
LDFLAGS=-Xlinker -rpath /opt/cuda/lib64 -L/opt/cuda/lib64 -lcudart -lexpat -lGL -lGLU -lglut
NCFLAGS=-m64 -I/opt/cuda/include -gencode arch=compute_20,code=sm_20 -gencode arch=compute_20,code=sm_21 -g

all: $(TARGET)

$(TARGET): $(OBJECTS)
	g++ $(FLAGS) $(OBJECTS) -o $(TARGET) $(LDFLAGS)

%.o: %.cpp
	$(CC) -o $@ -c $(CFLAGS) $<

%.co: %.cu
	$(NVCC) -o $@ -c $(NCFLAGS) $<

clean:
	rm -rf $(OBJECTS) layout
