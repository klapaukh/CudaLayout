#include <stdio.h>
#include <stdlib.h>
#include <expat.h>


int main(int,char**);


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
  
  //Finished reading the xml, so free the memory
  XML_ParserFree(p);

  return EXIT_SUCCESS;
}
