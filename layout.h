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
#define HOOKES_LAW_SPRING 1 << 0
#define LOG_SPRING 1 << 1
#define FRICTION 1 << 2
#define DRAG 1 << 3
#define COULOMBS_LAW 1 << 4
#define CHARGED_WALLS 1 << 5
#define DEGREE_BASED_CHARGE 1 << 6
#define CHARGED_EDGE_CENTERS 1 << 7
#define WRAP_AROUND_FORCES 1 << 8 


#endif
