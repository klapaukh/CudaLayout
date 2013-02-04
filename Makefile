OBJECTS=main.o layout.co graph.o graphmlReader.o
TARGET=layout
CC=g++
NVCC=/opt/cuda/bin/nvcc
CFLAGS=-pedantic -Wall -Wextra -lint -I/opt/cuda/include
LDFLAGS=-Xlinker -rpath /opt/cuda/lib64 -L/opt/cuda/lib64 -lcudart -lexpat -lGL -lGLU -lglut
NCFLAGS=-m64 -I/opt/cuda/include

all: $(TARGET)

$(TARGET): $(OBJECTS)
	g++ $(FLAGS) $(OBJECTS) -o $(TARGET) $(LDFLAGS)

%.o: %.cpp
	$(CC) -o $@ -c $(CFLAGS) $<

%.co: %.cu
	$(NVCC) -o $@ -c $(NCFLAGS) $<

clean:
	rm -rf $(OBJECTS) layout
