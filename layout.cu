#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <cuda_runtime.h>
#include <math_functions.h>

#include "layout.h"
#include "debug.h"

#define MAX_NODES 150

void handleError(cudaError_t, const char*);

__device__ float computeKineticEnergy(node* nodes, int numNodes, float mass) {
	float totalEk = 0;
	for (int i = 0; i < numNodes; i++) {
		//0.5*m*v^2
		float speed = nodes[i].dx * nodes[i].dx + nodes[i].dy * nodes[i].dy;
		speed = sqrt(speed);
		totalEk += 0.5 * mass * speed * speed;
	}
	return totalEk;
}

__global__ void layout(node* nodes_all, unsigned char* edges_all, int* numNodes_all, layout_params* paramsArg, float* finalEK, int numGraphs) {

	int graphIdx = blockIdx.x;
	int me = threadIdx.x;

	if (graphIdx > numGraphs) {
		return;
	}

	int nodes_start = 0;
	int edges_start = 0;
	for (int i = 0; i < graphIdx; i++) {
		int n = numNodes_all[i];
		nodes_start += n;
		edges_start += n * n;
	}

	int numNodes = numNodes_all[graphIdx];

	if (me >= numNodes) {
		return;
	}

	if(numNodes > MAX_NODES){
	//	printf("MAX_NODES too small %d\n", numNodes);
		return;
	}
	__shared__ node nodes[MAX_NODES];
	__shared__ layout_params params;
	__shared__ unsigned char edges[MAX_NODES * MAX_NODES];


	//Shared memory copy, don't need to double up work
	if(me == 0){
		params = *paramsArg;
	}
	nodes[me] = nodes_all[nodes_start + me];


	for (int i = 0; i < numNodes; i++) {
		edges[(me*numNodes) + i] = edges_all[edges_start + (me*numNodes) + i];
	}
	//Make sure the copies are visible to everyone
	__syncthreads();

	int forcemode = params.forcemode;
	float fx, fy;

	node node_me = nodes[me];

	if (((forcemode & (CHARGED_WALLS | BOUNCY_WALLS)) == 0) && me == 0) {
		//Keep the graph centered if there are no walls
		nodes[me].x = params.width / 2;
		nodes[me].y = params.height / 2;
		nodes[me].dx = 0;
		nodes[me].dy = 0;

		nodes_all[me].x = nodes[me].x;
		nodes_all[me].y = nodes[me].y;
		nodes_all[me].dx = nodes[me].dx;
		nodes_all[me].dy = nodes[me].dy;
		return;
	}
	for (int z = 0; z < params.iterations; z++) {
		fx = fy = 0;

		//Do node - node interactions
		for (int i = 0; i < numNodes; i++) {
			if (i == me) {
				continue;
			}

			node nodes_i = nodes[i];
			//Distance between two nodes
			float dx = node_me.x - nodes_i.x;
			float dy = node_me.y - nodes_i.y;
			float dist = sqrtf(dx * dx + dy * dy);

			//Work out the repulsive coulombs law force
			if ((forcemode & COULOMBS_LAW) != 0) {
				float q1 = node_me.charge, q2 = nodes[i].charge;

				if (dist < 1 || !isfinite(dist)) {
					dist = 1;
				}
				float f = (params.ke) * q1 * q2 / (dist * dist * dist);
				if (isfinite(f)) {
					fx += dx * f;
					fy += dy * f;
				}
			}

			//For nodes which are also connected
			if (edges[i + me * numNodes]) {
				if ((forcemode & HOOKES_LAW_SPRING) != 0) {
					//Attractive spring force
					float naturalWidth = nodes_i.width + node_me.width;
					float naturalHeight = nodes_i.height + node_me.height;
					float naturalLength = sqrt(naturalWidth * naturalWidth + naturalHeight * naturalHeight);
					float springF = -(params.kh) * (abs(dist) - naturalLength);
					fx += springF * dx / dist;
					fy += springF * dy / dist;
				} else if ((forcemode & LOG_SPRING) != 0) {
					float naturalWidth = nodes_i.width + node_me.width;
					float naturalHeight = nodes_i.height + node_me.height;
					float naturalLength = sqrt(naturalWidth * naturalWidth + naturalHeight * naturalHeight);
					float springF = (params.kl) * log(dist / naturalLength);
					fx += springF * dx / dist;
					fy += springF * dy / dist;
				}
			}
		}

		//Charged edges
		if ((forcemode & CHARGED_EDGE_CENTERS) != 0) {
			for (int src = 0; src < numNodes; src++) {
				for (int dst = src; dst < numNodes; dst++) {
					if (src != me && dst != me && edges[src + dst * numNodes]) {
						//Iterate through all the edges, but don't double up, skip non edges
						//And skip edges connected to me

						//Find the position of the edge center
						float edgex = (nodes[src].x + nodes[dst].x) / 2.0;
						float edgey = (nodes[src].y + nodes[dst].y) / 2.0;

						//----Edge - node ---//
						//Coulombs law it!
						float q1 = node_me.charge, q2 = params.edgeCharge;

						float dx = node_me.x - edgex;
						float dy = node_me.y - edgey;
						float dist = sqrtf(dx * dx + dy * dy);

						if (dist < 1 || !isfinite(dist)) {
							dist = 1;
						}

						float f = (params.ke) * q1 * q2 / (dist * dist * dist);
						if (isfinite(f)) {
							fx += dx * f;
							fy += dy * f;
						}

						//--- Edge - edges --//
						q1 = q2 = params.edgeCharge;
						//Go through all of my edges
						for (int end = 0; end < numNodes; end++) {
							if (edges[end + me * numNodes]) {
								//There is an edge between me and them!
								float edge2x = (node_me.x + nodes[end].x) / 2.0f;
								float edge2y = (node_me.y + nodes[end].y) / 2.0f;

								dx = edge2x - edgex;
								dy = edge2y - edgey;
								float dist = sqrtf(dx * dx + dy * dy);

								if (dist < 1 || !isfinite(dist)) {
									dist = 1;
								}

								float f = (params.ke) * q1 * q2 / (dist * dist * dist);
								if (isfinite(f)) {
									fx += dx * f;
									fy += dy * f;
								}
							}
						}
					}
				}
			}
		}

		//--------------General Update Actions--------------------------//

		//Charged walls if they are active
		if ((forcemode & CHARGED_WALLS) != 0) {
			float x = node_me.x;
			float y = node_me.y;
			float charge = (params.ke) * (params.kw) * node_me.charge;
			float forcexl = charge / ((x - 1) * (x - 1));
			float forceyt = charge / ((y - 1) * (y - 1));
			float forcexr = -charge / (((params.width) + 1 - x) * (params.width + 1 - x));
			float forceyb = -charge / ((params.height + 1 - y) * (params.height + 1 - y));
			fx += forcexl + forcexr;
			fy += forceyb + forceyt;
		}

		//Gravity well if it is active
		if ((forcemode & GRAVITY_WELL) != 0) {
			float dx = node_me.x - params.width / 2;
			float dy = node_me.y - params.height / 2;
			float dist = sqrt(dx * dx + dy * dy);
			if (dist < 100) {
				dist = 100;
			}
			// float gravForce = params->kg * params->mass * params->wellMass / (dist*dist);
			float gravForce = params.ke * node_me.charge * params.wellMass / (dist * dist);
			fx = gravForce * dx / dist;
			fy = gravForce * dy / dist;
		}

		//Friction against ground
		float g = 9.8f;
		float speed = node_me.dx * node_me.dx + node_me.dy * node_me.dy;
		speed = sqrt(speed);
		float normx = node_me.dx / speed;
		float normy = node_me.dy / speed;

		if ((forcemode & FRICTION) != 0) {
			if (node_me.dx == 0 && node_me.dy == 0) {
				//I am stationary
				float fFric = params.mus * params.mass * g;
				if (abs(fFric * normx) >= abs(fx) && abs(fFric * normx) >= abs(fy)) {
					fx = fy = 0;
				} else {
					//Just ignore friction this tick -- only really happens once
				}
			} else {
				float fFric = params.muk * params.mass * g;
				fx += -copysign(fFric * normx, node_me.dx);
				fy += -copysign(fFric * normy, node_me.dy);
			}
		}

		//Drag
		if ((forcemode & DRAG) != 0 && speed != 0) {
			float crossSec = node_me.width / 100.0f; //Conversion to m?
			float fDrag = 0.25f * crossSec * speed * speed;
			float fdx = -copysign(fDrag * normx, node_me.dx);
			float fdy = -copysign(fDrag * normy, node_me.dy);

			fx += fdx;
			fy += fdy;
		}

		//Move
		//F=ma => a = F/m
		float ax = fx / params.mass;
		float ay = fy / params.mass;

		if (!isfinite(ax)) {
			ax = 0;
		} else if (ax > params.width / 3) {
			ax = params.width / 3;
		} else if (ax < -(params.width) / 3) {
			ax = -(params.width) / 3;
		}

		if (!isfinite(ay)) {
			ay = 0;
		} else if (ay > params.height / 3) {
			ay = params.height / 3;
		} else if (ay < -(params.height) / 3) {
			ay = -(params.height) / 3;
		}

		node_me.nextX = node_me.x + node_me.dx * params.time;
		node_me.nextY = node_me.y + node_me.dy * params.time;
		node_me.nextdy = node_me.dy + ay * params.time;
		node_me.nextdx = node_me.dx + ax * params.time;

		//This part creates NaN values when it doesn't even get run!
		if ((forcemode & (CHARGED_WALLS | BOUNCY_WALLS)) != 0) {
			//Make sure it won't be travelling too fast
			if (node_me.nextdx * params.time > params.width / 2) {
				node_me.nextdx = params.width / (2 * params.time);
			} else if (node_me.nextdx * params.time < -(params.width) / 2) {
				node_me.nextdx = -(params.width) / (2 * params.time);
			}

			if (node_me.nextdy * params.time > params.height / 2) {
				node_me.nextdy = params.height / (2 * params.time);
			} else if (node_me.nextdy * params.time < -(params.height) / 2) {
				node_me.nextdy = -(params.height) / (2 * params.time);
			}

			//But wait - There are bounds to check!
			float collided = params.coefficientOfRestitution; //coeeficient of restitution
			if (node_me.nextX + node_me.width / 2 > params.width) {
				node_me.nextX = 2 * params.width - node_me.nextX - node_me.width;
				node_me.nextdx *= collided;
			} else if (node_me.nextX < node_me.width / 2) {
				node_me.nextX = node_me.width - node_me.nextX;
				node_me.nextdx *= collided;
			}

			if (node_me.nextY + node_me.height / 2 > params.height) {
				node_me.nextY = 2 * params.height - node_me.nextY - node_me.height;
				node_me.nextdy *= collided;
			} else if (node_me.nextY < node_me.height / 2) {
				node_me.nextY = node_me.height - node_me.nextY;
				node_me.nextdy *= collided;
			}
		}

		//Actually update the position of the nodes.
		node_me.x = node_me.nextX;
		node_me.y = node_me.nextY;
		node_me.dx = node_me.nextdx;
		node_me.dy = node_me.nextdy;

		nodes[me].x = node_me.x;
		nodes[me].y = node_me.y;
		nodes[me].dx = node_me.dx;
		nodes[me].dy = node_me.dy;

		__syncthreads();

	}

	//Clean up at the end
	nodes_all[nodes_start + me] = nodes[me];

	if (me == 1) {
		finalEK[graphIdx] = computeKineticEnergy(nodes, numNodes, params.mass);
	}
}

void graph_layout(graph** g, int numGraphs, layout_params* params) {

//Need to flatten all the host memory to make transfer fast
	float* finalEK_host = (float*) malloc(sizeof(float) * numGraphs); //This is written to straight from the GPu
	int* numNodes_host = (int*) malloc(sizeof(int) * numGraphs);
	if (finalEK_host == NULL || numNodes_host == NULL) {
		fprintf(stderr, "Failed to allocated memory for finalEK or numNodes");
		exit(-1);
	}

	int lengthEdges = 0;
	int lengthNodes = 0;
	int maxNumNodes = 0;

	for (int i = 0; i < numGraphs; i++) {
		numNodes_host[i] = g[i]->numNodes;
		finalEK_host[i] = -1;
		lengthEdges += numNodes_host[i] * numNodes_host[i];
		lengthNodes += numNodes_host[i];
		maxNumNodes = max(maxNumNodes, g[i]->numNodes);
	}

	node* nodes_host;
	unsigned char* edges_host;

	if (numGraphs == 1) {
		nodes_host = g[0]->nodes;
		edges_host = g[0]->edges;
	} else {

		//It would be best to linearise all the memory!
		edges_host = (unsigned char*) malloc(sizeof(unsigned char) * lengthEdges);
		nodes_host = (node*) malloc(sizeof(node) * lengthNodes);
		if (edges_host == NULL || nodes_host == NULL) {
			fprintf(stderr, "Failed to allocate memory for edges or nodes host");
			exit(-1);
		}

		//We can use one array to store all the nodes, and just give internal pointers to it around
		int nodes_offset = 0;
		int edges_offset = 0;
		for (int i = 0; i < numGraphs; i++) {
			int num_nodes = g[i]->numNodes;
			int num_edges = num_nodes * num_nodes;
			memcpy(nodes_host + nodes_offset, g[i]->nodes, sizeof(node) * num_nodes);
			memcpy(edges_host + edges_offset, g[i]->edges, sizeof(unsigned char) * num_edges);

			//Delete the old ones
			free(g[i]->nodes);
			free(g[i]->edges);
			//Put in the new values
			g[i]->nodes = nodes_host + nodes_offset;
			g[i]->edges = edges_host + edges_offset;

			nodes_offset += num_nodes;
			edges_offset += num_edges;
		}
	}

//	 need to allocate memory for nodes and edges on the device
	unsigned char* edges_device;
	node* nodes_device;
	float* finalEK_device;
	int* numNodes_device;

	cudaError_t err;
	layout_params* params_device;

#ifdef DEBUG
	cudaDeviceProp prop;
	int numDevices = -1;

	err = cudaGetDeviceCount(&numDevices);
	handleError(err, "Getting number of devices");

	printf("Found %d devices\n", numDevices);
	if (numDevices < 1) {
		exit(-1);
	}

	err = cudaGetDeviceProperties(&prop, 0);
	handleError(err,"Getting device properties");

	printf("Kernel time out enabled: %s\n", prop.kernelExecTimeoutEnabled?"true":"false");

	err = cudaDeviceReset();
	handleError(err, "Device Reset");

	err = cudaDeviceSynchronize();
	handleError(err, "Waiting for device reset finish");
#endif

	err = cudaMalloc(&edges_device, sizeof(unsigned char) * lengthEdges);
	handleError(err, "Allocating GPU memory for edges");

	err = cudaMalloc(&nodes_device, sizeof(node) * lengthNodes);
	handleError(err, "Allocating GPU memory for nodes");

	err = cudaMalloc(&numNodes_device, sizeof(int) * numGraphs);
	handleError(err, "Allocating GPU memory for number of nodes");


	err = cudaMalloc(&finalEK_device, sizeof(float) * numGraphs);
	handleError(err, "Allocating GPU memory for finalEK");

	err = cudaMalloc(&params_device, sizeof(layout_params));
	handleError(err, "Allocating GPU memory for params");

	/* copy data to device */
	err = cudaMemcpyAsync(edges_device, edges_host, sizeof(unsigned char) * lengthEdges, cudaMemcpyHostToDevice);
	handleError(err, "cudaMemcpy edges to device");

	err = cudaMemcpyAsync(nodes_device, nodes_host, sizeof(node) * lengthNodes, cudaMemcpyHostToDevice);
	handleError(err, " cudaMemcpy nodes to device");

	err = cudaMemcpyAsync(params_device, params, sizeof(layout_params), cudaMemcpyHostToDevice);
	handleError(err, "cudaMemcpy layout_params to device");

	err = cudaMemcpyAsync(numNodes_device, numNodes_host, sizeof(int) * numGraphs, cudaMemcpyHostToDevice);
	handleError(err, "cudaMemcpy numNodes to device");

	/*COMPUTE*/
	int nth = maxNumNodes;
	int nbl = numGraphs;

//	err = cudaDeviceSynchronize();
//	handleError(err, "Waiting for copy to device to finish");

	if (params->cpuLoop) {
		int iterations = params->iterations;
		params->iterations = 1;
		for (int i = 0; i < iterations; i++) {
			layout<<<nbl, nth>>>(nodes_device, edges_device, numNodes_device, params_device, finalEK_device, numGraphs);
		}
		params->iterations = iterations;
	} else {
		layout<<<nbl, nth>>>(nodes_device, edges_device, numNodes_device, params_device, finalEK_device, numGraphs);
	}
	err = cudaGetLastError();
	handleError(err,"launching kernel");
//	err = cudaDeviceSynchronize();
//	handleError(err, "Waiting for layout to finish");

	/*After computation you must copy the results back*/
	err = cudaMemcpyAsync(nodes_host, nodes_device, sizeof(node) * lengthNodes, cudaMemcpyDeviceToHost);
	handleError(err, "cudaMemcpy nodes to host");

	err = cudaMemcpyAsync(finalEK_host, finalEK_device, sizeof(float) * numGraphs, cudaMemcpyDeviceToHost);
	handleError(err, "cudaMemcpy nodes to host");

	for (int i = 0; i < numGraphs; i++) {
		g[i]->finalEK = finalEK_host[i];
	}

	free(finalEK_host);

	err = cudaDeviceSynchronize();
	handleError(err, "Waiting for everything to finish");
	/*
	 All finished, free the memory now
	 */
	err = cudaFree(nodes_device);
	handleError(err, "cudaFree nodes");

	err = cudaFree(edges_device);
	handleError(err, "cudaFree edges");

	err = cudaFree(params_device);
	handleError(err, "cudaFree layout_params");

	err = cudaFree(numNodes_device);
	handleError(err, "cudaFree numNodes_device");

	err = cudaFree(finalEK_device);
	handleError(err, "cudaFree finalEK_device");

}

void handleError(cudaError_t error, const char* context) {
	if (error != cudaSuccess) {
		printf("Cuda error occurred in: %s\n", context);
		printf("--%s\n", cudaGetErrorString(error));
		exit(-1);
	}
}

