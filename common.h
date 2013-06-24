/*
 * common.h
 *
 *  Created on: May 6, 2013
 *      Author: roma
 */

#ifndef COMMON_H_
#define COMMON_H_


typedef struct {
  int width;
  int height;
  int iterations;
  int forcemode;
  float ke;
  float kh;
  float kl;
  float kw;
  float mass;
  float time;
  float coefficientOfRestitution;
  float mus;
  float muk;
  float kg;
  float wellMass;
  float edgeCharge;
  bool cpuLoop;
  float finalKinectEnergy;
} layout_params;



#endif /* COMMON_H_ */
