#include <stdio.h>
#include <stdlib.h>

#include "graphmlReader.h"
#include "layout.h"

int main(int argc, char** argv){
  /*Check arguments to make sure you got a file*/
  if(argc != 2){
    printf("Usage: layout filename\n");
    return EXIT_FAILURE;
  }

  graph* g = read(argv[1]);
  if(g == NULL){
    printf("Creating a graph failed. Terminating");
    return EXIT_FAILURE;
  }
  
  graph_initRandom(g,80,80,1920,1080);
  /*The graph is now is a legal state. 
    It is possible to lay it out now
  */
  
  graph_layout(g,1920,1080);

  graph_toSVG(g, "test.svg");
  graph_free(g);
  return EXIT_SUCCESS;

  
}
