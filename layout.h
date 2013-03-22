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

typedef struct {
  int width;
  int height;
  int iterations;
  int forcemode;
  float ke;
  float kh;
  float kl;
  float kw;
  float mass;
  float time;
  float coefficientOfRestitution;
  float mus;
  float muk;
  float kg;
  float wellMass;
  float edgeCharge;
} layout_params;


void graph_layout(graph* g, layout_params* params);

//Force Modes 
#define HOOKES_LAW_SPRING 1 << 0
#define LOG_SPRING 1 << 1
#define FRICTION 1 << 2
#define DRAG 1 << 3
#define BOUNCY_WALLS 1 << 4
#define CHARGED_WALLS 1 << 5
#define GRAVITY_WELL 1 << 6
#define COULOMBS_LAW 1 << 7
#define DEGREE_BASED_CHARGE 1 << 8
#define CHARGED_EDGE_CENTERS 1 << 9
#define WRAP_AROUND_FORCES 1 << 10
#endif
