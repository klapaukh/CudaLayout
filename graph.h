#ifndef GRAPHH
#define GRAPHH

#include "common.h"

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
	unsigned char width;
	unsigned char height;
	unsigned char label;
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
	unsigned char* edges;
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
