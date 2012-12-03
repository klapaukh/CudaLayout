#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <float.h>

#include "graph.h"

graph* graph_create(void){
  /*Set up an empty graph*/
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
  free(g);
}

void graph_initRandom(graph* g,int width, int height, int screenWidth, int screenHeight){
  srand48(time(NULL));
  int i;
  for(i = 0; i < g->numNodes; i++){
    g->nodes[i].width = width;
    g->nodes[i].height = height;
    g->nodes[i].x = drand48() * (screenWidth-width) + width/2;
    g->nodes[i].y = drand48() * (screenHeight-height) + height/2;
    g->nodes[i].dx = 0;
    g->nodes[i].dy = 0;
  }
}

void graph_toSVG(graph* g, const char* filename, int screenwidth, int screenheight, bool hasWalls){
  FILE* svg = fopen(filename, "w");
  if(svg == NULL){
    printf("Failed to create file %s.\n",filename);
    return;
  }
  
  int stat;
  stat = fprintf(svg, "<?xml version=\"1.0\" encoding=\"ISO-8859-1\" standalone=\"no\"?>\n");
  stat = fprintf(svg, "<!DOCTYPE svg PUBLIC \"-//W3C//DTD SVG 20010904//EN\"\n");
  stat = fprintf(svg, "\"http://www.w3.org/TR/2001/REC-SVG-20010904/DTD/svg10.dtd\">\n");
  stat = fprintf(svg, "<svg xmlns=\"http://www.w3.org/2000/svg\"\n");
  stat = fprintf(svg, "xmlns:xlink=\"http://www.w3.org/1999/xlink\" xml:space=\"preserve\"\n");
  stat = fprintf(svg, "width=\"%dpx\" height=\"%dpx\"\n",screenwidth,screenheight);
  if(hasWalls){
    stat = fprintf(svg, "viewBox=\"0 0 %d %d\"\n", screenwidth,screenheight);
  }else{
    float minx= FLT_MAX, maxx= FLT_MIN, miny = FLT_MAX, maxy= FLT_MIN;
    for(int i=0 ;i < g ->numNodes;i++){
      node* n = g->nodes+i;
      if(n->x - n->width/2 < minx){
        minx = n->x - n->width/2;
      }
      if(n->x + n->width/2 > maxx){
        maxx = n->x + n->width/2;
      }
      if(n->y - n->height/2 < miny){
        miny = n->y - n->height/2;
      }
      if(n->y + n->height/2 > maxy){
        maxy = n->y + n->height/2;
      }
    }

    stat = fprintf(svg, "viewBox=\"%ld %ld %ld %ld\"\n", (long)minx, (long)miny, (long)(maxx-minx), (long)(maxy-miny));
  }
  stat = fprintf(svg, "zoomAndPan=\"disable\" >\n");


  int i,j;
  /*Draw edges*/
  for(i = 0; i < g->numNodes; i++){
    for(j = i+1; j < g->numNodes; j++){
      if(g->edges[i+j*g->numNodes]){
	int x1 = g->nodes[i].x;
	int x2 = g->nodes[j].x;
	int y1 = g->nodes[i].y;
	int y2 = g->nodes[j].y;
	stat = fprintf(svg, "<line x1=\"%d\" x2=\"%d\" y1=\"%d\" y2=\"%d\" stroke=\"%s\" fill=\"%s\" opacity=\"%.2f\"/>\n",
		       x1,x2,y1,y2,"rgb(255,0,0)","rgb(255,0,0)",1.0f);
	if(stat <0){
	  printf("An error occured while writing to the file");
	  fclose(svg);
	  return;
	}
      }
    }
  }

  /*Draw nodes*/
  for(i = 0; i < g->numNodes; i++){
    node* n = g->nodes+i;
    int x = (int)(n->x - n->width/2);
    int y = (int)(n->y - n->height/2);
    int width = n->width;
    int height = n->height;
    stat = fprintf(svg, "<rect x=\"%d\" y=\"%d\" width=\"%d\" height=\"%d\" fill=\"%s\" stroke=\"%s\" opacity=\"%.2f\"/>\n",
	    x,y,width,height, "rgb(0,0,255)","rgb(0,0,0)", 1.0f);
    if(stat < 0){
      printf("An error occured while writing to the file");
      fclose(svg);
      return;
    }
  }

  stat = fprintf(svg , "<rect x=\"0\" y=\"0\" width=\"%d\" height=\"%d\" stroke=\"rgb(0,255,0)\" fill-opacity=\"0\"/>\n",screenwidth, screenheight);

  fprintf(svg,"</svg>");
  fclose(svg);

}
