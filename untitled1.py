#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 18 18:04:32 2018

@author: casper
"""
from Utils import IO
import networkx as nx, numpy as np, matplotlib.pyplot as plt
dataDir = 'Psycho' # relative path careful
df    = IO.readCSV(f'{dataDir}/Graph_min1_1.csv', header = 0, index_col = 0)
h     = IO.readCSV(f'{dataDir}/External_min1_1.csv', header = 0, index_col = 0)
#
graph   = nx.from_pandas_adjacency(df)
#        for i, j in graph.edges():
#            graph[i][j]['weight'] *= 10
#
attr = {}
for node, row in h.iterrows():
    attr[node] = dict(H = row['externalField'], nudges = 0)
nx.set_node_attributes(graph, attr)
#graphs.append(graph)

sig = lambda x : 1 / (1 + np.exp(- 3*x))

fig, ax = plt.subplots()
colors = plt.cm.tab20(np.arange(graph.number_of_nodes()))
for idx, node in enumerate(graph.nodes()):
    e = abs(graph.nodes[node]['H'])
    for neighbor in graph.neighbors(node):
        e += abs(graph[node][neighbor]['weight'])
    ax.scatter(e, sig(e), label = node, color = colors[idx])
ax.legend()
fig.show()
        