#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <expat.h>
#include "graphmlReader.h"

//MAX_LEN is such that the index will fit into a 8 bits
#define BUFF_SIZE 1024
#define MAX_LEN 200
#define ID_LEN 6

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

void startTag(void* data, const char* element, const char** attributes){
  graphData* graph = (graphData*)data;

  //only actually care about 2 different kinds of object
  if(strcmp(element, "node")==0){
    for(int i=0; attributes[i] != NULL; i+=2){
      if(strcmp(attributes[i],"id") == 0){
	if(graph->numNode >= MAX_LEN){
	  printf("Too many nodes in file. Greater than MAX_LEN (%d)\n", MAX_LEN);
	  return;
	}
	//printf("Node: %s\n", attributes[i+1]);
	strncpy(graph->nodes[graph->numNode].id, attributes[i+1], ID_LEN);
	graph->numNode++;
      }
    }
  }else if(strcmp(element, "edge") == 0){
    if(graph->numEdge >= MAX_LEN){
      printf("Too many edges in file. Greater than MAX_LEN (%d)\n", MAX_LEN);
      return;
    }
    for(int i=0; attributes[i] != NULL; i+=2){
      if(strcmp(attributes[i],"id") == 0){
	//printf("%d: %s: %s\n", i, attributes[i], attributes[i+1]);
	strncpy(graph->edges[graph->numEdge].id, attributes[i+1], ID_LEN);
      }
      if(strcmp(attributes[i], "source")==0){
	//printf("%d: Source: %s\n", i, attributes[i+1]);
	strncpy(graph->edges[graph->numEdge].source, attributes[i+1], ID_LEN);
      }
      if(strcmp(attributes[i], "target")==0){
	//printf("%d: Target: %s\n", i, attributes[i+1]);
	strncpy(graph->edges[graph->numEdge].target, attributes[i+1], ID_LEN);
      }
    }
    graph->numEdge++;
  }
}

void endTag(void* data, const char* element){
  (void)data;
  (void)element;
}

graph* read(const char* filename){
  //We have a file to read, now lets try to read it
  XML_Parser p = XML_ParserCreate(NULL); //We do no specify the encoding
  if(p == NULL){
    printf("Allocating Memory for parser failed\n");
    return NULL;
  }

  XML_SetElementHandler(p, startTag, endTag);

  //Start reading the actual XML
  FILE* xmlFile = fopen(filename, "r");
  if(xmlFile == NULL){
    printf("XML file \"%s\" failed to open. Terminating\n", filename);
    return NULL;
  }

  char buff[BUFF_SIZE];
  int len = 10;

  //Set up the thing to pass around
  graphData data;
  data.numNode = 0;
  data.numEdge = 0;
  XML_SetUserData(p, &data);
  while(!feof(xmlFile)){
    len = fread(buff, 1, BUFF_SIZE, xmlFile);
    if(ferror(xmlFile)){
      printf("An error occured while trying to read the file.\n");
      fclose(xmlFile);
      return NULL;
    }

    //Successfully read something, time to parse, woo!
    if(!XML_Parse(p, buff, len, !len)){ // len == 0 => finished => need to negate
      fprintf(stderr, "Parse error at line %ld:\n%s\n", XML_GetCurrentLineNumber(p),
	      XML_ErrorString(XML_GetErrorCode(p)));
      return NULL;
    }
  }
  if(len !=0){
    XML_Parse(p, buff,0, 1); //It's definitely over
  }

  //Finished reading the xml, so free the memory
  fclose(xmlFile);
  XML_ParserFree(p);

  //data should now be sensible
  graph* g = graph_create();
  g->numNodes = data.numNode;
  g->numEdges = data.numEdge;
  g->numEdgeLabels = data.numEdge;
  g->numNodeLabels = data.numNode;
  g->nodeLabels = (char**)malloc(sizeof(char*)* data.numNode);
  g->edgeLabels = (char**)malloc(sizeof(char*)* (data.numEdge+1));
  g->nodes = (node*)malloc(sizeof(node) * data.numNode);
  g->edges = (unsigned char*)malloc(sizeof(unsigned char)*data.numNode * data.numNode);

  int i;
  g->edgeLabels[0] = (char*)malloc(sizeof(char));
  *(g->edgeLabels[0]) = '\0';
  for(i = 0; i< data.numEdge;i++){
    g->edgeLabels[i+1] = (char*)malloc(sizeof(char)* 6);
    strncpy(g->edgeLabels[i+1],data.edges[i].id,6);
  }
  for(i =0; i < data.numNode;i++){
    //Set the labels
    g->nodeLabels[i] = (char*)malloc(sizeof(char) * 6);
    strncpy(g->nodeLabels[i],data.nodes[i].id,6);

    //Initialise the nodes
    g->nodes[i].label = i;
  }

  //Initialise all edges to 0
  int j;
  for(i =0; i < data.numNode; i++){
    for(j=0; j < data.numNode; j++){
      g->edges[i+j*g->numNodes] = 0;
    }
  }

  //Find the edges which actually exist!
  for(i=0; i < data.numEdge;i++){
    int sourceid = -1;
    int targetid = -1;
    for(j=0;j < data.numNode;j++){
      if(strcmp(data.edges[i].source, data.nodes[j].id) == 0){
	sourceid = j;
      }
      if(strcmp(data.edges[i].target, data.nodes[j].id) == 0){
	targetid = j;
      }
    }
    if( sourceid == -1 || targetid == -1){
      printf("Could not find nodes for edge (%s,%s). Failed to create  graph\n",
	     data.edges[i].source, data.edges[i].target);
      return NULL;
    }
    g->edges[sourceid+targetid*g->numNodes] = i+1;
    g->edges[targetid+sourceid*g->numNodes] = i+1;
  }

  //Graph is now actually working
  printf("Number of Nodes: %d \nNumber of Edges: %d\n", g->numNodes, g->numEdges);
  return g;
}
