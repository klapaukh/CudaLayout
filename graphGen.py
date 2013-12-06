'
 Cuda Graph Layout Tool

 Copyright (C) 2013 Roman Klapaukh

 This program is free software: you can redistribute it and/or modify
 it under the terms of the GNU General Public License as published by
 the Free Software Foundation, either version 3 of the License, or
 (at your option) any later version.

 This program is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 GNU General Public License for more details.

 You should have received a copy of the GNU General Public License
 along with this program. If not, see <http://www.gnu.org/licenses/>.

'

from __future__ import print_function
import sys
import math


def pow(a, b):
    total = 1
    for i in range(b):
        total *= a
    return total

def circle(nnodes):
    for i in range(nnodes):
        print('  <node id="n{0}" />'.format(i))
        print('  <edge id="e{0}" source="n{0}" target="n{1}" />'.format(
            i, (i+1)%nnodes))
def k(n):
    for i in range(n):
        print('  <node id="n{0}" />'.format(i))
        for j in range(i+1,n):
            print('  <edge id="e{0}-{1}" source="n{0}" target="n{1}" />'.format(
                i, j))
def grid(n):
    for i in range(n):
        for j in range(n):
           print('	<node id="n{0}n{1}" />'.format(i,j))
           if i < n-1:
               print('	<edge id="e{0}n{1}--{2}n{1}" source="n{0}n{1}" target="n{2}n{1}" />'.format(i,j,i+1))
           if j < n-1:
               print('	<edge id="e{0}n{1}--{0}n{2}" source="n{0}n{1}" target="n{0}n{2}" />'.format(i,j,j+1))
def binaryTree(n):
    for i in range(n): 
        print(' <node id="n{0}" />'.format(i))
        parent = int(math.floor((i-1)/2))
        if parent >=0:
            print(' <edge id="e{0}--{1}" source="n{0}" target="n{1}" />'.format(i,parent))

nnodes = 5
if len(sys.argv) > 1:
    nnodes = int(sys.argv[2])
    if sys.argv[1] == '-c':
        gen = circle
    elif sys.argv[1] == '-k':
        gen = k
    elif sys.argv[1] == '-g':
        gen = grid
    elif sys.argv[1] == '-b':
        gen = binaryTree
    elif sys.argv[1] == '-bf':
        gen = binaryTree
        nnodes = int(pow(2,nnodes+1) - 1)
print('<?xml version="1.0"?>')
print('<graphml>')
print(' <graph id="g">')
gen(nnodes)
print(' </graph>')
print('</graphml>')
