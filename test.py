#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 11 08:55:35 2019

@author: casper
"""
<<<<<<< HEAD
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
=======

import networkx as nx
#import msgpack
#from numpy import *
from Models.fastIsing import Ising
#
g = nx.path_graph(5)
m = Ising(g)
#temps = linspace(0, 10)
#
#m = Ising(g)
#
#m.matchMagnetization(temps)

#fig, ax = subplots()
## dummy data
#N      = 100
#buffer = np.zeros((N, 2))
#p = ax.plot(*buffer.T, marker = '.')[0] # get plot object
#while True:
#    for idx in range(N):
#        buffer[idx, :] = np.random.rand(buffer.shape[1])
#        p.set_data(*buffer.T)
#        # recompute data limits
#        ax.relim()
#        ax.axes.autoscale_view(True, True, True)
#        
#        # update figure; flush events
#        fig.canvas.draw()
#        fig.canvas.flush_events()

>>>>>>> 65ca8ec635ca46bccf513a8b25177f2969076cf6
