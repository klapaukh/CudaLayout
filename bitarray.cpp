/*
 * bitarray.cpp
 *
 *  Created on: Aug 7, 2013
 *      Author: roma
 */

#include <stdlib.h>

#include "bitarray.h"
#include "debug.h"

int bitarray_numCells(int length){
	int numCells = length / CELL_SIZE;
	numCells += ((length % CELL_SIZE) == 0)? 0 : 1;
	return numCells;
}


bitarray bitarray_create(int length){
	int numCells = bitarray_numCells(length);
	return (bitarray)malloc(sizeof(unsigned char) * numCells);

}

void bitarray_free(bitarray array){
	free(array);
}

bool bitarray_get(bitarray array, int index){
	int cell = index / CELL_SIZE;
	int internalOffset = index % CELL_SIZE;
	unsigned char mask = 0x01 << internalOffset;
	unsigned char value = array[cell] & mask;
	return value;
}

void bitarray_set(bitarray array, int index, bool value){
	int cell = index / CELL_SIZE;
	int internalOffset = index % CELL_SIZE;
	unsigned char mask = 0x01 << internalOffset;
	if(value){
		array[cell] = array[cell] | mask;
	}else{
		mask = ~mask;
		array[cell] = array[cell] & mask;
	}
}

