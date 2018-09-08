# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import networkx as nx
from matplotlib.pyplot import *
from numpy import *
close('all')


i = nx.path_graph(4, nx.DiGraph())
j = nx.path_graph(2, nx.DiGraph())

graph = nx.DiGraph(); graph.add_node(0)
graph = nx.disjoint_union(j, graph)
rr  = hstack((arange(0, 5), arange(0,5) + .3))
pos = {node : (r, 0) if node <= 4 else (r, .5) for node, r in zip(graph.nodes(), rr)}
fig, ax = subplots(2,1); 
nx.draw(\
graph, pos = pos,  ax= ax[0], with_labels = True, \
node_size = 1000, node_color = 'w', font_size = 20, linewidths = 2)
ax[0].collections[0].set_edgecolor('k')
text(0,1, 'A', verticalalignment = 'center', horizontalalignment = 'center',\
     transform = ax[0].transAxes, fontdict = dict(fontsize = 20))

graph.add_edge(0, 5)
rr  = hstack((arange(0, 5), arange(0,5) + .3))
pos = {node : (r, 0) if node <= 4 else (r, .5) for node, r in zip(graph.nodes(), rr)}
nx.draw(\
graph, pos = pos,  ax= ax[1], with_labels = True, \
node_size = 1000, node_color = 'w', font_size = 20, linewidths = 2)
ax[1].collections[0].set_edgecolor('k')
text(0,1, 'B', verticalalignment = 'center', horizontalalignment = 'center',\
     transform = ax[1].transAxes, fontdict = dict(fontsize = 20))

fig.tight_layout()


# %%

graph = nx.path_graph(4, nx.DiGraph())
graph.add_edge(0, 4)
t = 'YABCX'
labs = {node : j for node, j in zip(graph.nodes, t)}
close('all')
pos = {node : (idx, 0) if node != 0 else (idx, .2) for idx, node in enumerate(graph.nodes())}
pos[4] = (1, .4)
fig, ax = subplots();
nx.draw(\
graph, pos = pos,  ax= ax, with_labels = True, \
node_size = 1000, node_color = 'w', font_size = 20, linewidths = 2, arrowsize = 40, labels = labs)
#nx.draw_networkx_nodes(graph, pos, label = graph.nodes(), node_size = 1100, node_color = 'white', linewidths = 1)
#nx.draw_networkx_labels(graph, pos, font_size = 20)
#nx.draw_networkx_edges(graph, pos, arrowstyle = '-|>', arrowsize = 20,ax = ax, arrows = True)
ax.axis('off')
ax.collections[0].set_edgecolor('k')
fig.tight_layout()
 