/*
* Cuda Graph Layout Tool
*
* Copyright (C) 2013 Roman Klapaukh
*
* This program is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* This program is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with this program. If not, see <http://www.gnu.org/licenses/>.
*
*/

#ifndef LAYOUTH
#define LAYOUTH

#include "graph.h"
#include "common.h"

void graph_layout(graph** g, int numGraphs, layout_params* params);

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
