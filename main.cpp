#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <GL/glut.h>

#include "graphmlReader.h"
#include "layout.h"


//GUI stuff
void display();
void idle();
void reshape(int,int);

void setLight();
void initCamera(int,int);

graph* glGraph = NULL;
int glWidth, glHeight, glIter;
float glKe, glKh, glTime;

int main(int argc, char** argv){
  /*Check arguments to make sure you got a file*/
  //There must be at least some arguments to get a file

  float ke = 500;
  float kh = 0.0005;
  char* filename = NULL;
  int swidth = 1920;
  int sheight = 1080;
  int iterations = 10000;
  bool gui = false;
  float time = 1;
  

  if(argc < 2){
    printf("Usage: layout [-f filename] [-gui] [-Ke 500] [-Kh 0.0005] [-i 10000] [-width 1920] [-height 1080] [-t 1]\n");
    return EXIT_FAILURE;
  }

  for(int i=1; i< argc; i++){
    if(strcmp(argv[i], "-f")==0){
      filename = argv[++i];
    }else if(strcmp(argv[i], "-Ke")==0){
      ke = atof(argv[++i]);
    }else if(strcmp(argv[i], "-Kh")==0){
      kh = atof(argv[++i]);
    }else if(strcmp(argv[i], "-i")==0){
      iterations = atoi(argv[++i]);
    }else if(strcmp(argv[i], "-width")==0){
      swidth = atoi(argv[++i]);
    }else if(strcmp(argv[i], "-height")==0){
      sheight = atoi(argv[++i]);
    }else if(strcmp(argv[i], "-gui")==0){
      gui = true;
    }else if(strcmp(argv[i], "-t")==0){
      time= atof(argv[++i]);

    }else{
      fprintf(stderr,"Unknown option %s\n",argv[i]);
      return EXIT_FAILURE;
    }
  }

  if(filename == NULL){
    perror("You must include a filename\n");
  }

  graph* g = read(filename);
  if(g == NULL){
    perror("Creating a graph failed. Terminating\n");
    return EXIT_FAILURE;
  }
 
  graph_initRandom(g,20,10,swidth,sheight);

  if(gui){
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA | GLUT_DEPTH);
    glutInitWindowSize(swidth, sheight);
    glutCreateWindow("Force Directed Layout");

    glutDisplayFunc(display);
    //glutReshapeFunc(reshape);
    glutIdleFunc(idle);

    setLight();
    initCamera(swidth,sheight);
    glGraph = g; 
    glWidth = swidth;
    glHeight = sheight;
    glKe = ke;
    glKh = kh;
    glIter = iterations;
    glTime = time;

    glutMainLoop();

  }



  /*The graph is now is a legal state. 
    It is possible to lay it out now
  */
  graph_toSVG(g, "before.svg", swidth, sheight);
  
  graph_layout(g,swidth,sheight,iterations, ke, kh, time);

  graph_toSVG(g, "after.svg",swidth,sheight);
  graph_free(g);
  return EXIT_SUCCESS;

}

void display(){
  glClearColor(1,1,1,1);
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
  glEnable(GL_DEPTH_TEST);
  glEnable(GL_LIGHTING);
  glEnable(GL_COLOR_MATERIAL);

  glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE);

  //draw edges
  glBegin(GL_LINES);
  glColor3f(1.0f, 0.0f, 0.0f); /* set object color as red */
  
  for(int i =0; i < glGraph->numNodes;i++){
    for(int j = i+1; j < glGraph->numNodes;j++){
      if(glGraph->edges[i+j*glGraph->numNodes]){
	float x1 = glGraph->nodes[i].x;
        float x2 = glGraph->nodes[j].x;
        float y1 = glGraph->nodes[i].y;
        float y2 = glGraph->nodes[j].y;
	glVertex2f(x1,y1);
	glVertex2f(x2,y2);
      }
    }
  }
  glEnd();

  //draw Nodes
  glBegin(GL_QUADS);
  glColor3f(0.0,0,1.0);

  for(int i = 0; i < glGraph->numNodes; i++){
    node* n = glGraph->nodes+i;
    int x = (int)(n->x - n->width/2);
    int y = (int)(n->y - n->height/2);
    int width = n->width;
    int height = n->height;
    glVertex2f(x,y);
    glVertex2f(x+width,y);
    glVertex2f(x+width,y+height);
    glVertex2f(x, y+height);
  } 
  glEnd();


  glDisable(GL_DEPTH_TEST);
  glDisable(GL_LIGHTING);
  glDisable(GL_COLOR_MATERIAL);

  GLenum error = glGetError();
  while( error != GL_NO_ERROR){
    fprintf(stderr, "%s\n", gluErrorString(error));
    error = glGetError();
  }
  
  glutSwapBuffers();
}

void idle(){
  graph_layout(glGraph,glWidth,glHeight,glIter, glKe, glKh, glTime);
  glutPostRedisplay();
}

void reshape(int w, int h){
  w=h=w;
}

void setLight(){
  float direction[] = { 0.0f, 0.0f, 1.0f, 0.0f };
  float diffintensity[] = { 0.7f, 0.7f, 0.7f, 1.0f };
  float ambient[] = { 0.2f, 0.2f, 0.2f, 1.0f };

  glLightfv(GL_LIGHT0, GL_POSITION, direction);
  glLightfv(GL_LIGHT0, GL_DIFFUSE, diffintensity);
  glLightfv(GL_LIGHT0, GL_AMBIENT, ambient);

  glEnable(GL_LIGHT0);
}

void initCamera(int width, int height){
  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  gluOrtho2D(0,width, height,0);
  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();
}
