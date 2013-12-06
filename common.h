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
} layout_params;

#endif /* COMMON_H_ */
