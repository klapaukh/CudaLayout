#include <stdio.h>
#include <stdlib.h>

int main(int argc, char** argv){
  //Check arguments to make sure you got a file
  if(argc != 2){
    printf("Usage: layout filename\n");
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
