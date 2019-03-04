#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 11 08:55:35 2019

@author: casper
"""

import networkx as nx
#import msgpack
#from numpy import *
from Models.fastIsing import Ising
import matplotlib.pyplot as plt
from Utils import plotting as plotz
g   = nx.barabasi_albert_graph(100, 2)
pos = nx.nx_agraph.graphviz_layout(g, 'neato')
g.add_edge(0,0)

fig, ax = plt.subplots()
plotz.addGraphPretty(g, ax = ax, positions = pos)
fig.show()

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
