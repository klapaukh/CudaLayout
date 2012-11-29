#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "graphmlReader.h"
#include "layout.h"

int main(int argc, char** argv){
  /*Check arguments to make sure you got a file*/
  //There must be at least some arguments to get a file

  float ke = 5;
  float kh = 5;
  char* filename = NULL;

  if(argc < 2 || argc % 2 == 0){
    printf("Usage: layout [-f filename] [-Ke 5] [-Kh 5]\n");
    return EXIT_FAILURE;
  }

  for(int i=1; i< argc; i+=2){
    if(strcmp(argv[i], "-f")==0){
      filename = argv[i+1];
    }else if(strcmp(argv[i], "-Ke")==0){
      ke = atof(argv[i+1]);
    }else if(strcmp(argv[i], "-Kh")==0){
      kh = atof(argv[i+1]);
    }else{
      fprintf(stderr,"Unknown option %s\n",argv[i]);
      return EXIT_FAILURE;
    }
  }

  if(filename == NULL){
    perror("You must include a filename\n");
  }

  graph* g = read(filename);
  if(g == NULL){
    perror("Creating a graph failed. Terminating\n");
    return EXIT_FAILURE;
  }
 
  int swidth = 1920;
  int sheight = 1080;
  
  graph_initRandom(g,20,10,swidth,sheight);
  /*The graph is now is a legal state. 
    It is possible to lay it out now
  */
  graph_toSVG(g, "before.svg", swidth, sheight);
  
  graph_layout(g,swidth,sheight,10000, ke, kh);

  graph_toSVG(g, "after.svg",swidth,sheight);
  graph_free(g);
  return EXIT_SUCCESS;

  
}
