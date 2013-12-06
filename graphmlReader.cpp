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

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <expat.h>
#include <sys/stat.h>
#include <dirent.h>
#include <errno.h>
#include <vector>

#include "graphmlReader.h"
#include "debug.h"

void parseDir(const char* dir, const char *internalDir, std::vector<graph*>&);

//MAX_LEN is such that the index will fit into a 8 bits
#define BUFF_SIZE 1024
#define MAX_LEN 500
#define ID_LEN 10

void startTag(void*, const char*, const char**);
void endTag(void*, const char*);

typedef struct edge {
	char id[ID_LEN];
	char source[ID_LEN];
	char target[ID_LEN];
} edge;

typedef struct tempnode {
	char id[ID_LEN];
} tempnode;

typedef struct graphData {
	int numNode;
	int numEdge;
	tempnode nodes[MAX_LEN];
	edge edges[MAX_LEN];
} graphData;

void startTag(void* data, const char* element, const char** attributes) {
	graphData* graph = (graphData*) data;

	//only actually care about 2 different kinds of object
	if (strcmp(element, "node") == 0) {
		for (int i = 0; attributes[i] != NULL; i += 2) {
			if (strcmp(attributes[i], "id") == 0) {
				if (graph->numNode >= MAX_LEN) {
					printf("Too many nodes in file (%d). Greater than MAX_LEN (%d)\n", graph->numNode, MAX_LEN);
					exit(-1);
				}
				//printf("Node: %s\n", attributes[i+1]);
				strncpy(graph->nodes[graph->numNode].id, attributes[i + 1], ID_LEN);
				graph->numNode++;
			}
		}
	} else if (strcmp(element, "edge") == 0) {
		if (graph->numEdge >= MAX_LEN) {
			printf("Too many edges in file (%d). Greater than MAX_LEN (%d)\n", graph->numEdge, MAX_LEN);
			exit(-1);
		}
		for (int i = 0; attributes[i] != NULL; i += 2) {
			if (strcmp(attributes[i], "id") == 0) {
				//debug("%d: %s: %s\n", i, attributes[i], attributes[i+1]);
				strncpy(graph->edges[graph->numEdge].id, attributes[i + 1], ID_LEN);
        graph->edges[graph->numEdge].id[ID_LEN-1] = '\0';  //Make sure there is a null!
			}
			if (strcmp(attributes[i], "source") == 0) {
				//debug("%d: Source: %s\n", i, attributes[i+1]);
				strncpy(graph->edges[graph->numEdge].source, attributes[i + 1], ID_LEN);
        graph->edges[graph->numEdge].source[ID_LEN-1] = '\0';  //Make sure there is a null!
			}
			if (strcmp(attributes[i], "target") == 0) {
				//debug("%d: Target: %s\n", i, attributes[i+1]);
				strncpy(graph->edges[graph->numEdge].target, attributes[i + 1], ID_LEN);
        graph->edges[graph->numEdge].target[ID_LEN-1] = '\0';  //Make sure there is a null!
			}
		}
		graph->numEdge++;
	}
}

void endTag(void* data, const char* element) {
	(void) data;
	(void) element;
}

graph* readFile(const char* filename) {
	//We have a file to read, now lets try to read it
	XML_Parser p = XML_ParserCreate(NULL); //We do no specify the encoding
	if (p == NULL) {
		printf("Allocating Memory for parser failed\n");
		return NULL;
	}

	XML_SetElementHandler(p, startTag, endTag);

	//Start reading the actual XML
	FILE* xmlFile = fopen(filename, "r");
	if (xmlFile == NULL) {
		printf("XML file \"%s\" failed to open.\n", filename);
		return NULL;
	}

	char buff[BUFF_SIZE];
	int len = 10;

	//Set up the thing to pass around
	graphData data;
	data.numNode = 0;
	data.numEdge = 0;
	XML_SetUserData(p, &data);
	while (!feof(xmlFile)) {
		len = fread(buff, 1, BUFF_SIZE, xmlFile);
		if (ferror(xmlFile)) {
			printf("An error occured while trying to read the file.\n");
			fclose(xmlFile);
			return NULL;
		}

		//Successfully read something, time to parse, woo!
		if (!XML_Parse(p, buff, len, !len)) { // len == 0 => finished => need to negate
			fprintf(stderr, "Parse error at line %ld:\n%s\n", XML_GetCurrentLineNumber(p), XML_ErrorString(XML_GetErrorCode(p)));
			return NULL;
		}
	}
	if (len != 0) {
		XML_Parse(p, buff, 0, 1); //It's definitely over
	}

	//Finished reading the xml, so free the memory
	fclose(xmlFile);
	XML_ParserFree(p);

	//data should now be sensible
	graph* g = graph_create();
	if(g == NULL){
		return NULL;
	}

	g->numNodes = data.numNode;
	g->numEdges = data.numEdge;
	g->numEdgeLabels = data.numEdge;
	g->numNodeLabels = data.numNode;
	g->nodeLabels = (char**) malloc(sizeof(char*) * data.numNode);
	g->edgeLabels = (char**) malloc(sizeof(char*) * (data.numEdge + 1));
	g->nodes = (node*) malloc(sizeof(node) * data.numNode);
	g->edges = bitarray_create(data.numNode * data.numNode);

	char* graphFileNamePointer = (char*) malloc(sizeof(char) * (strlen(filename) + 1)); //Space for null byte on the end
	if(graphFileNamePointer == NULL){
		fprintf(stderr,"Could not allocate graphFileNamePointer");
		return NULL;
	}

	strcpy(graphFileNamePointer, filename);
	g->filename = graphFileNamePointer;

	if (g->numNodes < 2) {
		printf("Not enough nodes in graph (%d < 2)", g->numNodes);
		return NULL;
	}

	int i;
	g->edgeLabels[0] = (char*) malloc(sizeof(char));
	*(g->edgeLabels[0]) = '\0';
	for (i = 0; i < data.numEdge; i++) {
		g->edgeLabels[i + 1] = (char*) malloc(sizeof(char) * 6);
		strncpy(g->edgeLabels[i + 1], data.edges[i].id, 6);
	}
	for (i = 0; i < data.numNode; i++) {
		//Set the labels
		g->nodeLabels[i] = (char*) malloc(sizeof(char) * 6);
		strncpy(g->nodeLabels[i], data.nodes[i].id, 6);

		//Initialise the nodes
		g->nodes[i].label = i;
	}

	//Initialise all edges to 0
	int j;
	for (i = 0; i < data.numNode; i++) {
		for (j = 0; j < data.numNode; j++) {
			bitarray_set(g->edges, i + j * g->numNodes, false);
		}
	}

	//Find the edges which actually exist!
	for (i = 0; i < data.numEdge; i++) {
		int sourceid = -1;
		int targetid = -1;
		for (j = 0; j < data.numNode; j++) {
			if (strcmp(data.edges[i].source, data.nodes[j].id) == 0) {
				sourceid = j;
			}
			if (strcmp(data.edges[i].target, data.nodes[j].id) == 0) {
				targetid = j;
			}
		}
		if (sourceid == -1 || targetid == -1) {
			printf("Could not find nodes for edge (%s,%s). Failed to create graph\n", data.edges[i].source, data.edges[i].target);
			return NULL;
		}
		bitarray_set(g->edges,sourceid + targetid * g->numNodes, true);
		bitarray_set(g->edges,targetid + sourceid * g->numNodes, true);
	}

	//Graph is now actually working
	debug("Number of Nodes: %d \nNumber of Edges: %d\n", g->numNodes, g->numEdges);
	return g;
}

graph** readDir(const char* dirname, int* numGraphs) {
	struct stat fileinfo;

	//Get the file information to make sure it is a directory
	if (stat(dirname, &fileinfo)) {
		//It failed :(
		int error = errno;
		fprintf(stderr, "Could not stat %s\n%s\nTerminating\n", dirname, strerror(error));
		return NULL;
	}

	//If it isn't a directory, give up!
	if ((fileinfo.st_mode & S_IFDIR) == 0) {
		fprintf(stderr, "%s is not a directory\n", dirname);
		return NULL;
	}

	std::vector<graph*> graphs;

	const char* initialDir = dirname[strlen(dirname) - 1] == '/' ? "" : "/";
	parseDir(dirname, initialDir, graphs);
	*numGraphs = graphs.size();

	debug("Found %d graphs\n", *numGraphs);
	if (graphs.size() == 0) {
		return NULL;
	}

	graph** graphArray = (graph**) malloc(sizeof(graph*) * graphs.size());
	if(graphArray == NULL){
		fprintf(stderr, "Could not allocate graph array\n");
		return NULL;
	}

	for (unsigned int i = 0; i < graphs.size(); i++) {
		graphArray[i] = graphs[i];
	}

	return graphArray;
}

void parseDir(const char* rootdir, const char * internalDir, std::vector<graph*>& graphs) {
	//Get a pointer
	char actualDir[BUFF_SIZE];

	if (strlen(rootdir) + strlen(internalDir) + 1 >= BUFF_SIZE) {
		fprintf(stderr, "Directory name too long (>%d) %s/%s\n", BUFF_SIZE, rootdir, internalDir);
	}
	strcpy(actualDir, rootdir);
	strcat(actualDir, internalDir);

	DIR* dir = opendir(actualDir);
	if (dir == NULL) {
		fprintf(stderr, "Could not open directory: %s\n", actualDir);
		return;
	}

	struct dirent* itemInDir = readdir(dir);
	while (itemInDir != NULL) {
		if (itemInDir->d_type == DT_DIR) {
			if (!(strcmp(itemInDir->d_name, ".") == 0 || strcmp(itemInDir->d_name, "..") == 0)) {
				debug("Dir: %s\n", itemInDir->d_name);

				if (strlen(internalDir) + 1 + strlen(itemInDir->d_name) < BUFF_SIZE) {
					char newInternalDir[BUFF_SIZE];
					strcpy(newInternalDir, internalDir);
					strcat(newInternalDir, itemInDir->d_name);
					strcat(newInternalDir, "/");

					parseDir(rootdir, newInternalDir, graphs);
				} else {
					fprintf(stderr, "File name too long %s / %s (> %d)\n", internalDir, itemInDir->d_name, BUFF_SIZE);
				}
			}
		} else if (strstr(itemInDir->d_name, ".graphml") != NULL) {
			char filenamebuff[BUFF_SIZE];
			strcpy(filenamebuff, actualDir);
			if (strlen(filenamebuff) + strlen(itemInDir->d_name) + 1 < BUFF_SIZE) { //Add null byte
				strcat(filenamebuff, itemInDir->d_name);

				debug("Reading file %s\n", filenamebuff);
				graph* g = readFile(filenamebuff);
        
        if(g != NULL){
				  if (g->filename != NULL) {
					  free(g->filename);
				  }
				  if(g->dir != NULL){
					  free(g->dir);
				  }

				  char* fnamep = (char*)malloc(sizeof(char)*strlen(itemInDir->d_name)+1); //Extra space for null byte
				  strcpy(fnamep, itemInDir->d_name);
				  g->filename = fnamep;

				  char* dirp = (char*)malloc(sizeof(char)*strlen(internalDir)+1); //Extra space for null byte
				  strcpy(dirp,internalDir);
				  g->dir = dirp;

					graphs.push_back(g);
				}
			} else {
				fprintf(stderr, "Filename too long (> %d) %s / %s\n", BUFF_SIZE, actualDir, itemInDir->d_name);
			}
		}
		itemInDir = readdir(dir);
	}

	if (closedir(dir)) {
		int error = errno;
		fprintf(stderr, "Failed to close directory\n%s\n", strerror(error));
	}
}
