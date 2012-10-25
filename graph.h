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

typedef struct graph{
  char** nodeLabels;
  char** edgeLabels;
  char*  edges;
  node*  nodes;
} graph;

void graph_free(graph*);

#endif
