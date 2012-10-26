#ifndef GRAPHH
#define GRAPHH

typedef struct node{
  float x;
  float y;
  float nextX;
  float nextY;
  float dx;
  float dy;
  float nextdx;
  float nextdy;
  float width;
  float height;
  int   label;
} node;

/*
  Edges is a array of indexes to the label. 0 means there is no edge.
  Note that this means that edgeLabels is 1 bigger than numEdges. 
 */
typedef struct graph{
  char** nodeLabels;
  char** edgeLabels;
  unsigned char*  edges;
  node*  nodes;
  int numNodeLabels;
  int numEdgeLabels;
  int numEdges;
  int numNodes;
} graph;

graph* graph_create(void);
void graph_free(graph*);
void graph_toSVG(graph*, char*);
void graph_initRandom(graph*, int, int, int, int);
void graph_layout(graph*,int,int);

#endif
