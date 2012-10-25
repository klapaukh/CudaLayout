#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <expat.h>

#define BUFF_SIZE 1024

int main(int,char**);
void startTag(void*, const char*, const char**);
void endTag(void*, const char*);



void startTag(void* data, const char* element, const char** attributes){
  (void)data;
  if(strcmp(element, "node")==0){
    int i=0;
    for(i=0; attributes[i] != NULL; i+=2){
      if(strcmp(attributes[i],"id") == 0){
	printf("Node: %s", attributes[i+1]);
      }
    }
  }else if(strcmp(element, "edge") == 0){
  }
}

void endTag(void* data, const char* element){
  (void)data;
  (void)element;
}
int main(int argc, char** argv){
  //Check arguments to make sure you got a file
  if(argc != 2){
    printf("Usage: layout filename\n");
    return EXIT_FAILURE;
  }

  //We have a file to read, now lets try to read it
  XML_Parser p = XML_ParserCreate(NULL); //We do no specify the encoding
  if(p == NULL){
    printf("Allocating Memory for parser failed\n");
    return EXIT_FAILURE;
  }
  
  XML_SetElementHandler(p, startTag, endTag);
  
  //Start reading the actual XML
  FILE* xmlFile = fopen(argv[1], "r");
  if(xmlFile == NULL){
    printf("XML file \"%s\" failed to open. Terminating\n", argv[1]);
    return EXIT_FAILURE;
  }
  
  char buff[BUFF_SIZE];
  int len = 10;
  
  while(!feof(xmlFile)){
    len = fread(buff, 1, BUFF_SIZE, xmlFile);
    if(ferror(xmlFile)){
      printf("An error occured while trying to read the file.\n");
      fclose(xmlFile);
      return EXIT_FAILURE;
    }
    
    //Successfully read something, time to parse, woo!
    XML_Parse(p, buff, len, !len); // len == 0 => finished => need to negate
  }
  if(len !=0){
    XML_Parse(p, buff,0, 1); //It's definitely over
  }
  
  
  //Finished reading the xml, so free the memory
  fclose(xmlFile);
  XML_ParserFree(p);

  return EXIT_SUCCESS;
}
