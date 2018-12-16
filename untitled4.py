#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 10 09:00:50 2018

@author: casper
"""

import networkx as nx, os
from numpy import *; from matplotlib.pyplot import *

tmp = 'weighted_person-person_projection_anonymous_combined.graphml'
fn  = f'{os.getcwd()}/Data/bn/{tmp}'

graph = nx.read_graphml(fn)
#giant = max(nx.connected_component_subgraphs(graph), key = len)

# %%
g = graph.copy()
theta = 40
for i in graph.nodes():
    if graph.degree(i) < theta:
        g.remove_node(i)
graph = g
# %%
fig, ax = subplots()
degs = dict(nx.degree(g))
ax.hist(degs.values())

# %%
import infcy, fastIsing

model = fastIsing.Ising(graph = graph, temperature = 0)
temps = linspace(0, 10)
