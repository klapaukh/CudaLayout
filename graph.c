#include <stdlib.h>
#include <stdio.h>
#include <time.h>

#include "graph.h"

void graph_layout(graph* g, int width, int height){
  /*
    need to allocate memory for nodes and edges on the device
  */
  unsigned char* edges_device;
  node* nodes_device;
  cudaError_t err;

  err = cudaMalloc(&edges_device, sizeof(unsigned char)* g->numNodes* g->numNodes);
  if(err != cudaSuccess){
    printf("Memory allocation for edges failed\n");
    return;
  }
  
  err = cudaMalloc(&nodes_device, sizeof(node) * g->numNodes);
  if(err != cudaSuccess){
    printf("Memory allocation for nodes failed\n");
    return;
  }
  
  /* copy data to device */
  err = cudaMemcpy(edges_device, g->edges, sizeof(unsigned char)* g->numNodes* g->numNodes, cudaMemcpyHostToDevice);
  if(err != cudaSuccess){
    printf("Error return from cudaMemcpy edges to device\n");
  }

  err = cudaMemcpy(nodes_device, g->nodes, sizeof(node)* g->numNodes, cudaMemcpyHostToDevice);
  if(err != cudaSuccess){
    printf("Error return from cudaMemcpy nodes to device\n");
  }

  

  /*After computation you must copy the results back*/
  err = cudaMemcpy(g->edges, edges_device, sizeof(unsigned char)* g->numNodes* g->numNodes, cudaMemcpyDeviceToHost);
  if(err != cudaSuccess){
    printf("Error return from cudaMemcpy edges to device\n");
  }

  err = cudaMemcpy(g->nodes, nodes_device, sizeof(node)* g->numNodes, cudaMemcpyDeviceToHost);
  if(err != cudaSuccess){
    printf("Error return from cudaMemcpy nodes to device\n");
  }
  
  
  
  /*
    All finished, free the memory now
  */
  err = cudaFree(nodes_device);
  if(err != cudaSuccess){
    printf("Freeing nodes failed\n");
  }
  
  err = cudaFree(edges_device);
  if(err != cudaSuccess){
    printf("Freeing edges failed\n");
  }
  
}

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
    g->nodes[i].x = drand48() * screenWidth;
    g->nodes[i].y = drand48() * screenHeight;
    g->nodes[i].dx = 0;
    g->nodes[i].dy = 0;
  }
}

void graph_toSVG(graph* g, char* filename){
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
  stat = fprintf(svg, "width=\"%dpx\" height=\"%dpx\"\n",1920,1080);
  stat = fprintf(svg, "viewBox=\"0 0 %d %d\"\n", 1920,1080);
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

  fprintf(svg,"</svg>");
  fclose(svg);

}
