#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 15:38:31 2019

@author: casper
"""
import matplotlib as mpl, matplotlib.pyplot as plt
from Models import  fastIsing, potts
import networkx as nx, numpy as np
from Utils import IO
from Toolbox import infcy
import time
from Utils.stats import KL, JS
deltas = 200
repeats = int(1e3)
start = time.time()

N = 5
#nudges[-1] = np.inf
graph = nx.barabasi_albert_graph(10, 2)
graph = nx.read_weighted_edgelist('Graphs/aves-barn-swallow-non-physical.edges')

#graph = nx.krackhardt_kite_graph()


dataDir = 'Graphs' # relative path careful
df    = IO.readCSV(f'{dataDir}/Graph_min1_1.csv', header = 0, index_col = 0)
h     = IO.readCSV(f'{dataDir}/External_min1_1.csv', header = 0, index_col = 0)
graph   = nx.from_pandas_adjacency(df)

graph = nx.Graph()
graph.add_edge(0,1)
graph.add_edge(1,2)
graph.add_edge(2,0)
graph[0][1]['weight'] = 10
#attr = {}


#for node, row in h.iterrows():
#    attr[node] = dict(H = row['externalField'], nudges = 0)
from scipy.ndimage import gaussian_filter1d

nudges = np.logspace(-1, 1, 10)
out = np.zeros((N, len(nudges), graph.number_of_nodes(), deltas))
for n in range(N):
    
    modelsettings = dict(\
                     graph = graph, \
                     updateType = 'single', \
                     nudgeType  = 'constant',\
                     magSide = 'neg')
    m = fastIsing.Ising(\
                    **modelsettings)
    temps = np.linspace(0, graph.number_of_nodes(), 100)
    samps = [m.matchMagnetization(temps, 100) for i in range(5)]

    samps = np.array(samps)
    samps = gaussian_filter1d(samps, 3, axis = -1)
    samps = samps.mean(0)
    mag, sus = samps
    snapshots    = infcy.getSnapShots(m, \
                                  nSamples = int(1e2), steps = int(1e3), \
                                  nThreads = -1)

#    IDX = np.argmin(abs(mag - .8 * mag.max()))
    IDX = np.argmax(sus)
    m.t = temps[IDX]
    conditional, px, mi = infcy.runMC(m, snapshots, deltas, repeats)
    
    
    for nidx, nudge in enumerate(nudges):
        for node, idx in m.mapping.items():
            m.nudges = {node : nudge}
            c, p, n_ = infcy.runMC(m, snapshots, deltas, repeats)
            out[n, nidx, idx, :] = KL(px, p).sum(-1)
print(time.time() - start)
# %%
#
fig, ax = plt.subplots()
ax.plot(temps, mag)
tax = ax.twinx()
tax.scatter(temps, sus)
ax.scatter(temps[IDX], mag[IDX], zorder = 1, color = 'red')
fig.show()


colors = plt.cm.tab20(np.linspace(0, 1, graph.number_of_nodes()))

a = np.trapz(out[..., deltas //  2 : ], axis = -1 )
# %%
b = np.trapz(out[..., deltas // 2:], axis = -1)

idx = np.argmax(a, axis = -1)


# %%
fig, (ax, ax1) = plt.subplots(1, 2, gridspec_kw = dict(width_ratios = [1, .44]))
tmp = set()

#nudges[-1] = int(100)
for gidx, maxNode in enumerate(idx):
    for jdx, nudge in enumerate(nudges):
        
        node = maxNode[jdx]
        ax.scatter(nudge, a[gidx, jdx, node], color = colors[node], alpha = 1)
        
        tmp.add(m.rmapping[node])
        
elements = [plt.Line2D(*[[0],[0]], marker = '.', color = colors[m.mapping[node]], label = node) for node in tmp]
#inax = ax.twinx()# ((1.05, .5, .25, .25))
ax.set(xlabel  = 'nudge size', ylabel = "largest causal impact $D_{kl}(P' || P)$")
nx.draw(graph, ax = ax1, with_labels = 1)
ax1.legend(handles = elements, title = 'node', bbox_to_anchor = (1.01, 1))
#ax.set_xlim(0, 20)
#ax.set_ylim(0, .05)
ax.set_xscale('log')
fig.show()

# %%
for kdx, nudge in enumerate(out):
    for jdx, ni in enumerate(nudge):
        fig, tax = plt.subplots()
        [tax.plot(i, color = colors[xdx]) for xdx, i in enumerate(ni)]
        tax.set_title(nudges[jdx])