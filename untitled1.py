#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  5 10:37:09 2018

@author: casper
"""

import networkx as nx
from matplotlib.pyplot import *
n = 10

graphs = [nx.barabasi_albert_graph(n, i) for i in range(1, n)]
i = 0
for graph in graphs:
    fig, ax = subplots()
    ax.hist(dict(graph.degree()).values())
    ax.set_title(i)
    i += 1