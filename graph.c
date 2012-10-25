#include <stdlib.h>

#include "graph.h"

graph* graph_create(void){
  //Set up an empty graph
  graph* g = (graph*)malloc(sizeof(graph));
  g->nodeLabels = NULL;
  g->edgeLabels = NULL;
  g->nodes = NULL;
  g->edges = NULL;
  g->numNodeLabels = 0;
  g->numEdgeLabels = 0;
  g->numEdges = 0;
  g->numNodes = 0;

  return g;
}

void graph_free(graph* g){
  if(g->nodeLabels != NULL){
    int i;
    for(i=0;i< g->numNodeLabels; i++){
      free(g->nodeLabels[i]);
    }
    free(g->nodeLabels);
  }
  if(g->edgeLabels != NULL){
    int i;
    for(i=0;i< g->numEdgeLabels+1; i++){
      free(g->edgeLabels[i]);
    }
    free(g->edgeLabels);
  }
  if(g->edges != NULL){
    free(g->edges);
  }
  if(g->nodes != NULL){
    free(g->nodes);
  }
}
