/*
* Cuda Graph Layout Tool
*
* Copyright (C) 2013 Roman Klapaukh
*
* This program is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* This program is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with this program. If not, see <http://www.gnu.org/licenses/>.
*
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

