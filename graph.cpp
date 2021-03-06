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

#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <float.h>
#include <math.h>
#include <errno.h>
#include <sys/stat.h>
#include <string.h>

#include "graph.h"

FILE* ensureFile(const char* filename);

graph* graph_create(void) {
	/*Set up an empty graph*/
	graph* g = (graph*) malloc(sizeof(graph));
	if(g == NULL){
		fprintf(stderr, "No memory to allocated graph\n");
		return NULL;
	}
	g->dir = NULL;
	g->filename = NULL;
	g->nodeLabels = NULL;
	g->edgeLabels = NULL;
	g->nodes = NULL;
	g->edges = NULL;
	g->numNodeLabels = 0;
	g->numEdgeLabels = 0;
	g->numEdges = 0;
	g->numNodes = 0;
	g->finalEK = -1;

	return g;
}

void graph_free(graph* g) {
	if (g->nodeLabels != NULL) {
		int i;
		for (i = 0; i < g->numNodeLabels; i++) {
			free(g->nodeLabels[i]);
		}
		free(g->nodeLabels);
	}
	if (g->edgeLabels != NULL) {
		int i;
		for (i = 0; i < g->numEdgeLabels + 1; i++) {
			free(g->edgeLabels[i]);
		}
		free(g->edgeLabels);
	}

	if(g -> dir != NULL) {
		free(g->dir);
	}
	if(g -> filename != NULL){
		free( g->filename);
	}
	free(g);
}

void graph_initRandom(graph* g, int width, int height, int screenWidth,
		int screenHeight, float nodeCharge) {
	srand48(time(NULL));
	int i;
	for (i = 0; i < g->numNodes; i++) {
		g->nodes[i].width = width;
		g->nodes[i].height = height;

		bool overlapping = true;
		while (overlapping) {
			g->nodes[i].x = drand48() * (screenWidth - width) + width / 2;
			g->nodes[i].y = drand48() * (screenHeight - height) + height / 2;

			overlapping = false;
			for(int j = 0 ; j < i ;j ++){
				if(abs(g->nodes[i].x - g->nodes[j].x) + abs(g->nodes[i].y - g->nodes[j].y) < 2){
					overlapping = true;
					break;
				}
			}
		}
		g->nodes[i].dx = 0;
		g->nodes[i].dy = 0;
		g->nodes[i].charge = nodeCharge;
	}
}

void graph_toSVG(graph* g, const char* filename, int screenwidth,
		int screenheight, bool hasWalls, long time, layout_params* params) {
	FILE* svg = ensureFile(filename);
	if (svg == NULL) {
		printf("Failed to create file (%s).\n", filename);
		return;
	}

	int stat;
	stat =
			fprintf(svg,
					"<?xml version=\"1.0\" encoding=\"ISO-8859-1\" standalone=\"no\"?>\n");
	stat = fprintf(svg,
			"<!DOCTYPE svg PUBLIC \"-//W3C//DTD SVG 20010904//EN\"\n");
	stat = fprintf(svg,
			"\"http://www.w3.org/TR/2001/REC-SVG-20010904/DTD/svg10.dtd\">\n");
	stat = fprintf(svg, "<svg xmlns=\"http://www.w3.org/2000/svg\"\n");
	stat =
			fprintf(svg,
					"xmlns:xlink=\"http://www.w3.org/1999/xlink\" xml:space=\"preserve\"\n");
	stat = fprintf(svg, "width=\"%dpx\" height=\"%dpx\"\n", screenwidth,
			screenheight);
	if (hasWalls) {
		stat = fprintf(svg, "viewBox=\"0 0 %d %d\"\n", screenwidth,
				screenheight);
	} else {
		float minx = FLT_MAX, maxx = FLT_MIN, miny = FLT_MAX, maxy = FLT_MIN;
		for (int i = 0; i < g->numNodes; i++) {
			node* n = g->nodes + i;
			if (n->x - n->width / 2 < minx) {
				minx = n->x - n->width / 2;
			}
			if (n->x + n->width / 2 > maxx) {
				maxx = n->x + n->width / 2;
			}
			if (n->y - n->height / 2 < miny) {
				miny = n->y - n->height / 2;
			}
			if (n->y + n->height / 2 > maxy) {
				maxy = n->y + n->height / 2;
			}
		}

		stat = fprintf(svg, "viewBox=\"%ld %ld %ld %ld\"\n", (long) minx,
				(long) miny, (long) (maxx - minx), (long) (maxy - miny));
	}
	stat = fprintf(svg, "zoomAndPan=\"disable\" >\n");

	/**
	 * As a comment, print out the graph and data
	 */

	stat = fprintf(svg, "<!--\n"); // Begin comment block (for easy extraction)
	stat = fprintf(svg, "elapsed: %ld\n", time);
	stat = fprintf(svg, "filename: %s\n", g->filename);


	//Print the program arguments
	stat = fprintf(svg, "width: %d\n", params->width);
	stat = fprintf(svg, "height: %d\n", params->height);
	stat = fprintf(svg, "iterations: %d\n", params->iterations);
	stat = fprintf(svg, "forcemode: %d\n", params->forcemode);
	stat = fprintf(svg, "ke: %f\n", params->ke);
	stat = fprintf(svg, "kh: %f\n", params->kh);
	stat = fprintf(svg, "kl: %f\n", params->kl);
	stat = fprintf(svg, "kw: %f\n", params->kw);
	stat = fprintf(svg, "mass: %f\n", params->mass);
	stat = fprintf(svg, "time: %f\n", params->time);
	stat = fprintf(svg, "coefficientOfRestitution: %f\n",
			params->coefficientOfRestitution);
	stat = fprintf(svg, "mus: %f\n", params->mus);
	stat = fprintf(svg, "muk: %f\n", params->muk);
	stat = fprintf(svg, "kg: %f\n", params->kg);
	stat = fprintf(svg, "wellMass: %f\n", params->wellMass);
	stat = fprintf(svg, "edgeCharge: %f\n", params->edgeCharge);
	stat = fprintf(svg, "finalKineticEnergy: %f\n", g->finalEK);
	stat = fprintf(svg, "nodeWidth: %f\n", g->nodes[0].width);
	stat = fprintf(svg, "nodeHeight: %f\n", g->nodes[0].height);
	stat = fprintf(svg, "nodeCharge: %f\n", g->nodes[0].charge);


	stat = fprintf(svg, "-\n"); // Begin comment block (for easy extraction)

	//Print the graph as the adjacency matrix
	stat = fprintf(svg, "Start Graph:\n");
	stat = fprintf(svg, "%d\n", g->numNodes); // num Nodes
	for (int i = 0; i < g->numNodes; i++) {
		stat = fprintf(svg, "%0.2f %0.2f ", g->nodes[i].x, g->nodes[i].y);
		for (int j = 0; j < g->numNodes; j++) {
			stat = fprintf(svg, "%d ", (bitarray_get(g->edges,i * g->numNodes + j)) ? 1 : 0);
		}
		fprintf(svg, "\n"); //end row
	}

	stat = fprintf(svg, "-->\n"); // End comment block (for easy extraction)

	int i, j;
	/*Draw edges*/
	for (i = 0; i < g->numNodes; i++) {
		for (j = i + 1; j < g->numNodes; j++) {
			if (bitarray_get(g->edges,i + j * g->numNodes)) {
				int x1 = g->nodes[i].x;
				int x2 = g->nodes[j].x;
				int y1 = g->nodes[i].y;
				int y2 = g->nodes[j].y;
				stat =
						fprintf(svg,
								"<line x1=\"%d\" x2=\"%d\" y1=\"%d\" y2=\"%d\" stroke=\"%s\" fill=\"%s\" opacity=\"%.2f\"/>\n",
								x1, x2, y1, y2, "rgb(255,0,0)", "rgb(255,0,0)",
								1.0f);
				if (stat < 0) {
					printf("An error occured while writing to the file");
					fclose(svg);
					return;
				}
			}
		}
	}

	/*Draw nodes*/
	for (i = 0; i < g->numNodes; i++) {
		node* n = g->nodes + i;
		int x = (int) (n->x - n->width / 2);
		int y = (int) (n->y - n->height / 2);
		int width = n->width;
		int height = n->height;
		stat =
				fprintf(svg,
						"<rect x=\"%d\" y=\"%d\" width=\"%d\" height=\"%d\" fill=\"%s\" stroke=\"%s\" opacity=\"%.2f\"/>\n",
						x, y, width, height, "rgb(0,0,255)", "rgb(0,0,0)",
						1.0f);
		if (stat < 0) {
			printf("An error occurred while writing to the file");
			fclose(svg);
			return;
		}
	}

	stat =
			fprintf(svg,
					"<rect x=\"0\" y=\"0\" width=\"%d\" height=\"%d\" stroke=\"rgb(0,255,0)\" fill-opacity=\"0\"/>\n",
					screenwidth, screenheight);

	fprintf(svg, "</svg>");
	fclose(svg);

}

FILE* ensureFile(const char* filename){
	FILE* file =  fopen(filename, "w");
	if(file != NULL){
		return file;
	}

	//Ok lets check to see if the directory exists;
	char* dir = (char*)malloc(sizeof(char)*(strlen(filename)+3));
	if(dir == NULL){
		fprintf(stderr,"Failed to allocated space for dir\n");
		exit(-1);
	}
	strcpy(dir,filename);

	char* p = strrchr(dir,'/');
	if(p == NULL){
		return NULL;
	}
	*p = '\0';

	struct stat fileinfo;
	//Get the file information to make sure it is a directory
	int slashCount= 0;
	while (stat(dir, &fileinfo)) {
		//It failed, hopefully it just doesn't exist
		int error = errno;
		if(error != ENOENT){
			fprintf(stderr,"An error occurred while trying to create the file: %s\n%s\n",filename,strerror(error));
			return NULL;
		}

		p = strrchr(dir,'/');
		if(p == NULL){
			fprintf(stderr,"An error occurred while trying to create the file: %s\n%s\n",filename,strerror(error));
			return NULL;
		}
		*p = '\0';

		slashCount++;
	}

	for(;slashCount > 0 ; slashCount--){
		int end = strlen(dir);
		dir[end] = '/';
		if(mkdir(dir,S_IRUSR | S_IWUSR | S_IXUSR)){
			int error = errno;
			fprintf(stderr, "An error occurred while trying to create directories\n%s\n",strerror(error));
			return NULL;
		}
	}
	free(dir);
	return fopen(filename, "w");
}
