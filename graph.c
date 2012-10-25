#include <stdlib.h>

#include "graph.h"

void graph_free(graph* g){
  if(g->nodeLabels != NULL){
  }
  if(g->edgeLabels != NULL){
  }
  if(g->edges != NULL){
    free(g->edges);
  }
  if(g->nodes != NULL){
    free(g->nodes);
  }
}
