#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <expat.h>
#include "graphmlReader.h"

#define BUFF_SIZE 1024
#define MAX_LEN 300
#define ID_LEN 6

typedef struct edge {
  char id[ID_LEN];
  char source[ID_LEN];
  char target[ID_LEN];
} edge;

typedef struct node {
  char id[ID_LEN];
} node;

typedef struct graphData {
  int numNode;
  int numEdge;
  node nodes[MAX_LEN];
  edge edges[MAX_LEN];
} graphData;

void startTag(void* data, const char* element, const char** attributes){
  graphData* graph = (graphData*)data;

  //only actually care about 2 different kinds of object
  if(strcmp(element, "node")==0){
    int i=0;
    for(i=0; attributes[i] != NULL; i+=2){
      if(strcmp(attributes[i],"id") == 0){
	if(graph->numNode >= MAX_LEN){
	  printf("Too many nodes in file. Greater than MAX_LEN (%d)\n", MAX_LEN);
	  return;
	}
	printf("Node: %s\n", attributes[i+1]);
	strncpy(graph->nodes[graph->numNode].id, attributes[i+1], ID_LEN);
	graph->numNode++;
      }
    }
  }else if(strcmp(element, "edge") == 0){
    int i=0;
    if(graph->numEdge >= MAX_LEN){
      printf("Too many edges in file. Greater than MAX_LEN (%d)\n", MAX_LEN);
      return;
    }
    for(i=0; attributes[i] != NULL; i+=2){
      if(strcmp(attributes[i],"id") == 0){
	printf("Edge: %s\n", attributes[i+1]);
	strncpy(graph->edges[graph->numEdge].id, attributes[i+1], ID_LEN);
      }
      if(strcmp(attributes[i], "source")){
	printf("Source: %s\n", attributes[i+1]);
	strncpy(graph->edges[graph->numEdge].source, attributes[i+1], ID_LEN);
      }
      if(strcmp(attributes[i], "target")){
	printf("Target: %s\n", attributes[i+1]);
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

int read(char* filename){
  //We have a file to read, now lets try to read it
  XML_Parser p = XML_ParserCreate(NULL); //We do no specify the encoding
  if(p == NULL){
    printf("Allocating Memory for parser failed\n");
    return EXIT_FAILURE;
  }
  
  XML_SetElementHandler(p, startTag, endTag);
  
  //Start reading the actual XML
  FILE* xmlFile = fopen(filename, "r");
  if(xmlFile == NULL){
    printf("XML file \"%s\" failed to open. Terminating\n", filename);
    return EXIT_FAILURE;
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
      return EXIT_FAILURE;
    }
    
    //Successfully read something, time to parse, woo!
    if(!XML_Parse(p, buff, len, !len)){ // len == 0 => finished => need to negate
      fprintf(stderr, "Parse error at line %ld:\n%s\n", XML_GetCurrentLineNumber(p), 
	      XML_ErrorString(XML_GetErrorCode(p)));
      return 0;
    }
  }
  if(len !=0){
    XML_Parse(p, buff,0, 1); //It's definitely over
  }
  
  //Finished reading the xml, so free the memory
  fclose(xmlFile);
  XML_ParserFree(p);

  return 0;
}
