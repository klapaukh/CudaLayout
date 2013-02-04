#include <stdio.h>
#include <stdlib.h>

#include "layout.h"

__global__ void layout(node* nodes, unsigned char* edges, int numNodes, layout_params* params){
  int me = blockIdx.x * 8 + threadIdx.x;
  int forcemode = params-> forcemode;
  
  if(me >= numNodes){
    return;
  }
  float fx, fy;
  for(int z=0;z<params->iterations;z++){
    fx = fy = 0;
    for(int i =0; i < numNodes; i++){
      if( i == me){
	continue;
      }

      //Distance between two nodes
      float dx = nodes[me].x - nodes[i].x;
      float dy = nodes[me].y - nodes[i].y;
      float dist = sqrtf(dx*dx + dy *dy);

      //Work out the repulsive coulombs law force      
      if((forcemode & COULOMBS_LAW) != 0){
	float q1 = nodes[me].charge, q2 = nodes[i].charge;
	
	if(dist < 1 || !isfinite(dist)){
	  dist = 1;
	}
	float f = (params->ke)*q1*q2/ (dist*dist * dist);
	if(isfinite(f)){
	  fx += dx * f;
	  fy += dy * f;
	}
      }
      if((forcemode & HOOKES_LAW_SPRING) != 0 && edges[i + me * numNodes]){
	//Attractive spring force
	float naturalWidth = nodes[i].width + nodes[me].width;
	float naturalHeight = nodes[i].height + nodes[me].height;
	float naturalLength = sqrt(naturalWidth * naturalWidth + naturalHeight*naturalHeight);
	float springF = -(params->kh) * (abs(dist) - naturalLength);
	fx += springF * dx/dist;
	fy += springF * dy/dist;      
      }else if((forcemode & LOG_SPRING) != 0 && edges[i + me * numNodes]){
        float naturalWidth = nodes[i].width + nodes[me].width;
        float naturalHeight = nodes[i].height + nodes[me].height;
        float naturalLength = sqrt(naturalWidth * naturalWidth + naturalHeight*naturalHeight);
	float springF = (params->kl)* log(dist/ naturalLength);
	fx += springF * dx/dist;
        fy += springF * dy/dist;   
      }
    }


    //Charged walls if they are active
    if((forcemode & CHARGED_WALLS) != 0 ){
      float x = nodes[me].x;
      float y = nodes[me].y;
      float charge = (params->ke) * (params->kw) * nodes[me].charge;
      float forcexl = charge / ((x-1) *(x-1));
      float forceyt = charge / ((y-1) * (y-1)); 
      float forcexr = -charge / (((params->width)+1-x) *(params->width+1 - x));
      float forceyb = -charge / ((params->height+1 - y) *(params->height+1 - y));   
      fx += forcexl + forcexr;
      fy += forceyb + forceyt;
    }

    //Gravity well if it is active
    if((forcemode & GRAVITY_WELL) != 0){
      float dx = nodes[me].x - params->width/2;
      float dy = nodes[me].y - params->height/2;
      float dist = sqrt(dx*dx + dy*dy);
      if(dist < 100){
	dist = 100;
      }
      // float gravForce = params->kg * params->mass * params->wellMass / (dist*dist);
      float gravForce = params->ke * nodes[me].charge * params->wellMass / (dist*dist);
      fx = gravForce * dx / dist;
      fy = gravForce * dy / dist;
    }

    //Friction against ground
    float g = 9.8f;
    float speed = nodes[me].dx * nodes[me].dx + nodes[me].dy * nodes[me].dy;
    speed = sqrt(speed);
    float normx = nodes[me].dx / speed;
    float normy = nodes[me].dy / speed;
    
    if((forcemode & FRICTION) != 0){
      if(nodes[me].dx == 0 && nodes[me].dy ==0 ){
	//I am stationary
	float fFric = params->mus * params->mass * g;
	if(abs(fFric* normx) >= abs(fx)  && abs(fFric*normx) >= abs(fy)){
	  fx = fy = 0;
      }else{
	  //Just ignore friction this tick -- only really happens once
	}
      }else{
	float fFric = params->muk * params->mass * g;
	fx += -copysign(fFric*normx, nodes[me].dx);
	fy += -copysign(fFric*normy, nodes[me].dy);
      }
    }
    
    //Drag
    if((forcemode & DRAG)!= 0 &&  speed != 0){
      float crossSec = nodes[me].width / 100.0f; //Conversion to m?
      float fDrag = 0.25f * crossSec * speed * speed;
      float fdx = -copysign(fDrag * normx, nodes[me].dx);
      float fdy = -copysign(fDrag * normy, nodes[me].dy);
      
      fx += fdx;
      fy += fdy;
    }

    //Move
    //F=ma => a = F/m
    float ax = fx / params->mass;
    float ay = fy / params->mass;


    if((forcemode & (CHARGED_WALLS | BOUNCY_WALLS)) != 0){
      if(ax > params->width/3){
	ax = params->width/3;
      }else if(ax < -(params->width)/3){
	ax = -(params->width)/3;
      }else if(!isfinite(ax)){
	ax = 0;
      }
      
      if(ay > params->height/3){
	ay = params->height/3;
      }else if(ay < -(params->height)/3){
	ay = -(params->height)/3;
      }else if(!isfinite(ay)){
	ay = 0;
      }
    }
    
    nodes[me].nextX = nodes[me].x + nodes[me].dx*params->time;
    nodes[me].nextY = nodes[me].y + nodes[me].dy*params->time;
    nodes[me].nextdy =nodes[me].dy + ay*params->time;
    nodes[me].nextdx =nodes[me].dx + ax*params->time;
    

    if((forcemode & (CHARGED_WALLS | BOUNCY_WALLS)) != 0){
      //Make sure it won't be travelling too fast
      if(nodes[me].nextdx * params->time > params->width/2){
	nodes[me].nextdx = params->width/(2*params->time);
      }else if(nodes[me].nextdx * params->time < -(params->width)/2){
	nodes[me].nextdx = -(params->width)/(2*params->time);
      }
      
      if(nodes[me].nextdy * params->time > params->height / 2){
	nodes[me].nextdy = params->height/(2*params->time);
      }else if(nodes[me].nextdy * params->time < -(params->height)/2){
	nodes[me].nextdy = -(params->height)/(2*params->time);
      }
      
      //But wait - There are bounds to check!
      float collided = params->coefficientOfRestitution; //coeeficient of restitution
      if(nodes[me].nextX + nodes[me].width/2 > params->width){
	nodes[me].nextX = 2* params->width - nodes[me].nextX - nodes[me].width;
	nodes[me].nextdx *= collided;
      }else if(nodes[me].nextX < nodes[me].width/2){
	nodes[me].nextX =  nodes[me].width - nodes[me].nextX;
	nodes[me].nextdx *= collided;
      }
      
      if(nodes[me].nextY + nodes[me].height/2 > params->height){
	nodes[me].nextY = 2*params->height - nodes[me].nextY - nodes[me].height;
	nodes[me].nextdy *= collided;
      }else if(nodes[me].nextY < nodes[me].height/2){
	nodes[me].nextY = nodes[me].height - nodes[me].nextY; 
	nodes[me].nextdy *= collided;
      }
    }

    //Update
    nodes[me].x = nodes[me].nextX;
    nodes[me].y = nodes[me].nextY;
    nodes[me].dx = nodes[me].nextdx;
    nodes[me].dy = nodes[me].nextdy;
  }
}


void graph_layout(graph* g, layout_params* params){
  /*
    need to allocate memory for nodes and edges on the device
  */
  unsigned char* edges_device;
  node* nodes_device;
  cudaError_t err;
  layout_params* params_device;

  err = cudaMalloc(&edges_device, sizeof(unsigned char)* g->numNodes* g->numNodes);
  if(err != cudaSuccess){
    printf("Memory allocation for edges failed on GPU\n");
    return;
  }
  
  err = cudaMalloc(&nodes_device, sizeof(node) * g->numNodes);
  if(err != cudaSuccess){
    printf("Memory allocation for nodes failed on GPU\n");
    return;
  }

  err = cudaMalloc(&params_device, sizeof(layout_params));
  if(err != cudaSuccess){
    printf("Memory allocation for params failed on GPU\n");
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

  err = cudaMemcpy(params_device, params, sizeof(layout_params), cudaMemcpyHostToDevice);
  if(err != cudaSuccess){
    printf("Error return from cudaMemcpy params to device\n");
  }
  
  /*COMPUTE*/
  int nth = 8;
  int nbl = ceil(g->numNodes / 8.0);
  //printf("Graph has %d nodes with %d blocks and %d threads\n", g->numNodes, nbl, nth);
  layout<<<nbl,nth>>>(nodes_device, edges_device, g->numNodes,params_device);
  
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
  
  err = cudaFree(params_device);
  if(err != cudaSuccess){
    printf("Freeing params failed\n");
  }
}

