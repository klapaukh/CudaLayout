#include <stdio.h>
#include <stdlib.h>

#include "graphmlReader.h"

int main(int argc, char** argv){
  //Check arguments to make sure you got a file
  if(argc != 2){
    printf("Usage: layout filename\n");
    return EXIT_FAILURE;
  }

  graph* graph = read(argv[1]);
  if(graph == NULL){
    printf("Creating a graph failed. Terminating");
    return EXIT_FAILURE;
  }

  graph_free(graph);
  return EXIT_SUCCESS;

  
}
