#include <stdio.h>
#include <stdlib.h>

#include "layout.h"

__global__ void layout(node* nodes, unsigned char* edges, int width, int height){
  int idx = threadIdx.x;
  nodes[idx].x = width;
}


//extern "C" 
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

  
  /*COMPUTE*/
  int nt = g->numNodes;
  layout<<<1,nt>>>(nodes_device, edges_device,width,height);
  
  /*After computation you must copy the results back*/
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

