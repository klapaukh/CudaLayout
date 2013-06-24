#ifndef LAYOUTH
#define LAYOUTH

#include "graph.h"
#include "common.h"

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
