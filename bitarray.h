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
