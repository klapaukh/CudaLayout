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
 
  int swidth = 1920;
  int sheight = 1080;
  
  graph_initRandom(g,20,10,swidth,sheight);
  /*The graph is now is a legal state. 
    It is possible to lay it out now
  */
  graph_toSVG(g, "before.svg", swidth, sheight);
  
  graph_layout(g,swidth,sheight,10000);

  graph_toSVG(g, "after.svg",swidth,sheight);
  graph_free(g);
  return EXIT_SUCCESS;

  
}
