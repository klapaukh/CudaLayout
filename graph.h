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

#ifndef GRAPHH
#define GRAPHH

#include "common.h"
#include "bitarray.h"

typedef struct node {
	float x;
	float y;
	float nextX;
	float nextY;
	float dx;
	float dy;
	float nextdx;
	float nextdy;
	float charge;
	float width;
	float height;
	int label;
} node;

/*
 Edges is a array of indexes to the label. 0 means there is no edge.
 Note that this means that edgeLabels is 1 bigger than numEdges.
 */
typedef struct graph {
	char* dir;
	char * filename;
	char** nodeLabels;
	char** edgeLabels;
	bitarray edges;
	node* nodes;
	int numNodeLabels;
	int numEdgeLabels;
	int numEdges;
	int numNodes;
	float finalEK;
} graph;

graph* graph_create(void);
void graph_free(graph*);
void graph_toSVG(graph*, const char* outfile, int, int, bool, long time,
		layout_params* params);
void graph_initRandom(graph*, int, int, int, int, float);

#endif
