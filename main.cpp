#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <float.h>
#include <errno.h>

#include <GL/glut.h>

#include "graphmlReader.h"
#include "layout.h"
#include "statistics.h"

//GUI stuff
void display();
void idle();
void reshape(int, int);

void setLight();
void initCamera(int, int);

graph* glGraph = NULL;
layout_params* glParams = NULL;

void compute_graph_statistics(graph*, layout_params*);

void usage() {
	fprintf(stderr,
			"Usage: layout [-f filename] [-gui] [-noOuput] [-Ke 500] [-Kh 0.0005] [-Kl -0.05] [-nodeCharge 3] [-edgeCharge 3] [-Kg 0.06] [-mus 0.3] [-muk 0.04] [-i 10000] [-width 1920] [-height 1080] [-t 1] [-nm 1] [-m 1] [-cRest -0.9] [-friction 3] [-spring 1] [-walls 1] [-forces 1]\n");
	fprintf(stderr, "Forces:\n");

	fprintf(stderr, "\nFriction:\n");
	fprintf(stderr, " Friction            - 1\n");
	fprintf(stderr, " Drag                - 2\n");

	fprintf(stderr, "\nSpring:\n");
	fprintf(stderr, " Hooke's Law         - 1\n");
	fprintf(stderr, " Log Law             - 2\n");

	fprintf(stderr, "\nWalls:\n");
	fprintf(stderr, " Bouncy Walls        - 1\n");
	fprintf(stderr, " Charged Walls       - 2\n");
	fprintf(stderr, " Gravity Well        - 4\n");

	fprintf(stderr, "\nPrimary:\n");
	fprintf(stderr, " Coulombs Law        - 1\n");
	fprintf(stderr, " Degree-Based Charge - 2\n");
	fprintf(stderr, " Charged Edges       - 4\n");
	fprintf(stderr, " Wrap Around Forces  - 8\n");

}

int readInt(int argc, char** argv, int i) {
	if (i >= argc) {
		fprintf(stderr, "An int was not provided to %s\n", argv[i - 1]);
		exit (EXIT_FAILURE);
	}

	if (errno != 0) {
		perror("Something went wrong before readInt started");
	}
	errno = 0;
	char* strend;
	long val = strtol(argv[i], &strend, 10);

	/* Check for various possible errors */
	if ((errno == ERANGE && (val == LONG_MAX || val == LONG_MIN)) || (errno != 0 && val == 0)) {
		fprintf(stderr, "An error occured while parsing the argument to %s\n", argv[i - 1]);
		perror("readInt");
		exit (EXIT_FAILURE);
	}

	if (strend == argv[i]) {
		fprintf(stderr, "No digits were found for argument %s\n", argv[i - 1]);
		exit (EXIT_FAILURE);
	}

	if (val > INT_MAX || val < INT_MIN) {
		fprintf(stderr, "Value given to argument %s outside of integer range", argv[i - 1]);
		exit (EXIT_FAILURE);
	}

	/* If we got here, strtol() successfully parsed a number */
	if (*strend != '\0') { /* Not necessarily an error... */
		fprintf(stderr, "Further characters after number: %s in argument for %s\n", strend, argv[i - 1]);
	}

	return (int) val;
}

float readFloat(int argc, char** argv, int i) {
	if (i >= argc) {
		fprintf(stderr, "A float was not provided to %s\n", argv[i - 1]);
		exit (EXIT_FAILURE);
	}

	if (errno != 0) {
		perror("Something went wrong before readFloat started");
	}
	errno = 0;
	char* strend;
	float val = strtof(argv[i], &strend);

	/* Check for various possible errors */
	if ((errno == ERANGE && (val == FLT_MAX || val == FLT_MIN)) || (errno != 0 && val == 0)) {
		fprintf(stderr, "An error occured while parsing the argument to %s\n", argv[i - 1]);
		perror("readFloat");
		exit (EXIT_FAILURE);
	}

	if (strend == argv[i]) {
		fprintf(stderr, "No digits were found for argument %s\n", argv[i - 1]);
		exit (EXIT_FAILURE);
	}

	/* If we got here, strtol() successfully parsed a number */
	if (*strend != '\0') { /* Not necessarily an error... */
		fprintf(stderr, "Further characters after number: %s in argument for %s\n", strend, argv[i - 1]);
	}

	return val;
}

const char* readString(int argc, char** argv, int i) {
	if (i >= argc) {
		fprintf(stderr, "A String was not provided to %s\n", argv[i - 1]);
		exit (EXIT_FAILURE);
	}

	return argv[i];
}

int main(int argc, char** argv) {
	/*Check arguments to make sure you got a file*/
	//There must be at least some arguments to get a file
	const char* filename = NULL;
	bool gui = false;
	float nodeCharge = 3;
	bool output = true;

	layout_params* params = (layout_params*) malloc(sizeof(layout_params));
	params->ke = 500;
	params->kh = 0.0005;
	params->kl = -0.05;
	params->width = 1920;
	params->height = 1080;
	params->iterations = 10000;
	params->mass = 1;
	params->time = 1;
	params->coefficientOfRestitution = -0.9;
	params->mus = 0.2;
	params->muk = 0.04;
	params->kw = 3;
	params->kg = 0.06;
	params->wellMass = 1;
	params->edgeCharge = nodeCharge;
	params->forcemode = COULOMBS_LAW | HOOKES_LAW_SPRING | FRICTION | DRAG | BOUNCY_WALLS;

	if (argc < 2) {
		usage();
		return EXIT_FAILURE;
	}

	for (int i = 1; i < argc; i++) {
		if (strcmp(argv[i], "-f") == 0) {
			filename = readString(argc, argv, ++i);
		} else if (strcmp(argv[i], "-Ke") == 0) {
			params->ke = readFloat(argc, argv, ++i);
		} else if (strcmp(argv[i], "-Kh") == 0) {
			params->kh = readFloat(argc, argv, ++i);
		} else if (strcmp(argv[i], "-Kl") == 0) {
			params->kl = readFloat(argc, argv, ++i);
		} else if (strcmp(argv[i], "-nodeCharge") == 0) {
			nodeCharge = readFloat(argc, argv, ++i);
		} else if (strcmp(argv[i], "-edgeCharge") == 0) {
			params->edgeCharge = readFloat(argc, argv, ++i);
		} else if (strcmp(argv[i], "-Kg") == 0) {
			params->kg = readFloat(argc, argv, ++i);
		} else if (strcmp(argv[i], "-nm") == 0) {
			params->wellMass = readFloat(argc, argv, ++i);
		} else if (strcmp(argv[i], "-mus") == 0) {
			params->mus = readFloat(argc, argv, ++i);
		} else if (strcmp(argv[i], "-muk") == 0) {
			params->muk = readFloat(argc, argv, ++i);
		} else if (strcmp(argv[i], "-i") == 0) {
			params->iterations = readInt(argc, argv, ++i);
		} else if (strcmp(argv[i], "-width") == 0) {
			params->width = readInt(argc, argv, ++i);
		} else if (strcmp(argv[i], "-height") == 0) {
			params->height = readInt(argc, argv, ++i);
		} else if (strcmp(argv[i], "-gui") == 0) {
			gui = true;
		} else if (strcmp(argv[i], "-noOutput") == 0) {
			output = false;
		} else if (strcmp(argv[i], "-t") == 0) {
			params->time = readFloat(argc, argv, ++i);
		} else if (strcmp(argv[i], "-m") == 0) {
			params->mass = readFloat(argc, argv, ++i);
		} else if (strcmp(argv[i], "-cRest") == 0) {
			params->coefficientOfRestitution = readFloat(argc, argv, ++i);
		} else if (strcmp(argv[i], "-friction") == 0) {
			int fricForce = readInt(argc, argv, ++i);
			params->forcemode = params->forcemode & ~(FRICTION | DRAG);
			params->forcemode = params->forcemode | (fricForce << 2);
		} else if (strcmp(argv[i], "-spring") == 0) {
			int springForce = readInt(argc, argv, ++i);
			params->forcemode = params->forcemode & ~(HOOKES_LAW_SPRING | LOG_SPRING);
			params->forcemode = params->forcemode | (springForce);
		} else if (strcmp(argv[i], "-walls") == 0) {
			int wallForce = readInt(argc, argv, ++i);
			params->forcemode = params->forcemode & ~(BOUNCY_WALLS | CHARGED_WALLS | GRAVITY_WELL);
			params->forcemode = params->forcemode | (wallForce << 4);
		} else if (strcmp(argv[i], "-forces") == 0) {
			int primForce = readInt(argc, argv, ++i);
			params->forcemode = params->forcemode & ~(COULOMBS_LAW | DEGREE_BASED_CHARGE | CHARGED_EDGE_CENTERS | WRAP_AROUND_FORCES);
			params->forcemode = params->forcemode | (primForce << 7);

		} else {
			fprintf(stderr, "Unknown option %s\n", argv[i]);
			return EXIT_FAILURE;
		}
	}

	if (filename == NULL) {
		fprintf(stderr, "You must include a filename\n");
		usage();
		return EXIT_FAILURE;
	}

	graph* g = read(filename);
	if (g == NULL) {
		fprintf(stderr, "Creating a graph failed. Terminating\n");
		usage();
		return EXIT_FAILURE;
	}

	graph_initRandom(g, 20, 10, params->width, params->height, nodeCharge);

	if (gui) {
		glutInit(&argc, argv);
		glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA | GLUT_DEPTH);
		glutInitWindowSize(params->width, params->height);
		glutCreateWindow("Force Directed Layout");

		glutDisplayFunc(display);
		//glutReshapeFunc(reshape);
		glutIdleFunc(idle);

		setLight();
		initCamera(params->width, params->height);
		glGraph = g;
		glParams = params;
		glutMainLoop();

	}

	/*The graph is now is a legal state.
	 * It is not using the GUI.
	 It is possible to lay it out now
	 */
	if (output) {
		graph_toSVG(g, "before.svg", params->width, params->height, (params->forcemode & (BOUNCY_WALLS | CHARGED_WALLS)) != 0);
	}
	graph_layout(g, params);

	compute_graph_statistics(g, params);
	if (output) {
		graph_toSVG(g, "after.svg", params->width, params->height, (params->forcemode & (BOUNCY_WALLS | CHARGED_WALLS)) != 0);
	}
	graph_free(g);
	return EXIT_SUCCESS;

}

void display() {
	glClearColor(1, 1, 1, 1);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glEnable (GL_DEPTH_TEST);
	glEnable (GL_LIGHTING);
	glEnable (GL_COLOR_MATERIAL);

	glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE);

	if ((glParams->forcemode & (CHARGED_WALLS | BOUNCY_WALLS)) == 0) {
		//I actually need to normalise all the values for drawing
		float minx = FLT_MAX, maxx = FLT_MIN, miny = FLT_MAX, maxy = FLT_MIN;
		for (int i = 0; i < glGraph->numNodes; i++) {
			node* n = glGraph->nodes + i;
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
		glMatrixMode (GL_PROJECTION);
		glLoadIdentity();
		gluOrtho2D(minx, maxx, maxy, miny);
		glMatrixMode (GL_MODELVIEW);

	}

	//draw edges
	glBegin (GL_LINES);
	glColor3f(1.0f, 0.0f, 0.0f); /* set object color as red */

	for (int i = 0; i < glGraph->numNodes; i++) {
		for (int j = i + 1; j < glGraph->numNodes; j++) {
			if (glGraph->edges[i + j * glGraph->numNodes]) {
				float x1 = glGraph->nodes[i].x;
				float x2 = glGraph->nodes[j].x;
				float y1 = glGraph->nodes[i].y;
				float y2 = glGraph->nodes[j].y;
				glVertex2f(x1, y1);
				glVertex2f(x2, y2);
			}
		}
	}
	glEnd();

	//draw Nodes
	glBegin (GL_QUADS);
	glColor3f(0.0, 0, 1.0);

	for (int i = 0; i < glGraph->numNodes; i++) {
		node* n = glGraph->nodes + i;
		int x = (int) (n->x - n->width / 2);
		int y = (int) (n->y - n->height / 2);
		int width = n->width;
		int height = n->height;
		glVertex2f(x, y);
		glVertex2f(x + width, y);
		glVertex2f(x + width, y + height);
		glVertex2f(x, y + height);
	}
	glEnd();

	glDisable(GL_DEPTH_TEST);
	glDisable(GL_LIGHTING);
	glDisable(GL_COLOR_MATERIAL);

	GLenum error = glGetError();
	while (error != GL_NO_ERROR) {
		fprintf(stderr, "%s\n", gluErrorString(error));
		error = glGetError();
	}

	glutSwapBuffers();
}

void idle() {
	graph_layout(glGraph, glParams);
	glutPostRedisplay();
}

void reshape(int w, int h) {
	w = h = w;
}

void setLight() {
	float direction[] = { 0.0f, 0.0f, 1.0f, 0.0f };
	float diffintensity[] = { 0.7f, 0.7f, 0.7f, 1.0f };
	float ambient[] = { 0.2f, 0.2f, 0.2f, 1.0f };

	glLightfv(GL_LIGHT0, GL_POSITION, direction);
	glLightfv(GL_LIGHT0, GL_DIFFUSE, diffintensity);
	glLightfv(GL_LIGHT0, GL_AMBIENT, ambient);

	glEnable (GL_LIGHT0);
}

void initCamera(int width, int height) {
	glMatrixMode (GL_PROJECTION);
	glLoadIdentity();
	gluOrtho2D(0, width, height, 0);
	glMatrixMode (GL_MODELVIEW);
	glLoadIdentity();
}

void compute_graph_statistics(graph* g, layout_params* params) {

}
