OBJECTS=main.o layout.co graph.o graphmlReader.o
TARGET=layout
CC=g++
NVCC=nvcc
CFLAGS=-pedantic -Wall -Wextra -lint -I/opt/cuda50/include
LDFLAGS=-Xlinker -rpath /opt/cuda50/lib64 -L/opt/cuda50/lib64 -lcudart -lexpat
NCFLAGS=-m64 -I/opt/cuda50/include

all: $(TARGET)

$(TARGET): $(OBJECTS)
	g++ $(FLAGS) $(OBJECTS) -o $(TARGET) $(LDFLAGS)

%.o: %.cpp
	$(CC) -o $@ -c $(CFLAGS) $<

%.co: %.cu
	$(NVCC) -o $@ -c $(NCFLAGS) $<

clean:
	rm -rf $(OBJECTS) layout
