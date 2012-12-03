#ifndef LAYOUTH
#define LAYOUTH

#include "graph.h"

/**
Graph layout takes a graph
width of plane
height of plane
number of interations
ke
kh
mass
time
forceMode
 */
void graph_layout(graph* g, int width, int height, int iterations, float ke, float kh, float mass, float time, float coefficientOfResititution, int forcemode);

//Force Modes 
#define COULOMBS_LAW 1
#define HOOKES_LAW_SPRING 1 << 1
#define LOG_SPRING 1 << 2
#define FRICTION 1 << 3
#define DRAG 1 << 4

#endif
