#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 11 08:55:35 2019

@author: casper
"""

#import networkx as nx
#import msgpack
#from numpy import *
#from Models.fastIsing import Ising
#
#g = nx.path_graph(5)
#temps = linspace(0, 10)
#
#m = Ising(g)
#
#m.matchMagnetization(temps)

import numpy as np
height = np.array(\
                  [67, 67, 55, 65, 65, 65, 61, 58, 40, 40,\
                   58, 53, 59, 63, 51, 57, 43, 65, 45, 65,\
                   61, 58, 47, 58, 65, 74, 64, 28, 61, 46, 39])
height.sort()
threshold = 300
groupings = np.where(np.diff(height.cumsum() // threshold))[0]
ends      = np.hstack((groupings, height.size))
starts    = np.roll(ends.copy(), 1) + 1
starts[0] = 0

for start, end in zip(starts, ends):
    print(f'Grouping: {height[start:end]} sum: {height[start:end].sum()}')