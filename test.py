#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 11 08:55:35 2019

@author: casper
"""
#
#import networkx as nx
#import msgpack
#from numpy import *
#from matplotlib.pyplot import *
#from Models.fastIsing import Ising
#
#g = nx.barabasi_albert_graph(20, 10)
#temps = linspace(0, 100, 100)
#
#m = Ising(g)
#
#mags, sus = m.matchMagnetization(temps)
#
#fig, ax = subplots()
#ax.plot(temps, mags, temps, sus)
#fig.show()
#show()


import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull, Delaunay

x = np.linspace(-np.pi, np.pi)
p = np.random.rand(100, 2)
p = np.array([np.cos(x), np.sin(x)]).T
hull = ConvexHull(p)
fig, ax = plt.subplots()
ax.scatter(*p.T, color = 'red')
ax.plot(*p[hull.simplices].T)
fig.show()