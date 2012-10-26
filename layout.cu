#include <stdio.h>
#include <stdlib.h>

#include "layout.h"

__global__ void layout(node* nodes, unsigned char* edges, int numNodes, int width, int height, int interations){
  int me = threadIdx.x;
  
  float fx, fy;
  for(int i =0; i < numNodes; i++){
    fx = fy = 0;
    //Work out the repulsive coulombs law force
    float dx = nodes[i].x - nodes[me].x;
    float dy = nodes[i].y - nodes[me].y;
    float dist = sqrtf(dx*dx + dy *dy);

    float ke = 50000;
    float q1 = 3, q2 = 3;
    float f = ke*q1*q2/ (dist*dist);
     
    fx = dx * f;
    fy = dx * y;
    
    if(edges[i + me * numNodes]){
      //Attractive spring force
      //      float naturalDistance = nodes[i].width + nodes[me].height; //TODO different sizes
      float naturalWidth = nodes[i].width;
      float naturalHeight = nodes[i].height;
      float f = 0.2;
      fx += (-f) * (dx - naturalWidth);
      fy += (-f) * (dy - naturalHeight);      
    }
    //Move
    //F=ma => a = F/m
    float mass = 2;
    float ax = fx / mass;
    float ay = fy / mass;
    
    nodes[me].nextdy =nodes[me].dy + ay;
    nodes[me].nextdx =nodes[me].dx + ax;
    nodes[me].nextX = nodes[me].x + dx;
    nodes[me].nextY = nodes[me].y + dy;
    //Update
  }

}


void graph_layout(graph* g, int width, int height, int iterations){
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
  layout<<<1,nt>>>(nodes_device, edges_device, g->numNodes,width,height, iterations);
  
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

