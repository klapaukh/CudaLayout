/*
 * bitarray.h
 *
 *  Created on: Aug 7, 2013
 *      Author: roma
 */

#ifndef BITARRAY_H_
#define BITARRAY_H_

typedef unsigned char* bitarray;

#define CELL_SIZE 8

int bitarray_numCells(int length);
bitarray bitarray_create(int length);
void bitarray_free(bitarray array);
bool bitarray_get(bitarray array, int index);
void bitarray_set(bitarray array, int index, bool value);


#endif /* BITARRAY_H_ */
