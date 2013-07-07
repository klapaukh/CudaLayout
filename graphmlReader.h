#ifndef GRAPHMLREADERH
#define GRAPHMLREADERH

#include "graph.h"

graph* readFile(const char*);
graph** readDir(const char*, int*);

#endif
