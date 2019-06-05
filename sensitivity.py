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





deltas = 20
repeats = int(1e3)
start = time.time()

N = 20

out = np.zeros((N, len(nudges), m.nNodes, deltas))
nudges = np.logspace(-1, 2, 10)
nudges[-1] = np.inf
graph = nx.erdos_renyi_graph(5, .3)
graph = nx.path_graph(3)
for n in range(N):
    
    modelsettings = dict(\
                     graph = graph, \
                     updateType = 'single', \
                     nudgeType  = 'constant',\
                     magSide = '')
    m = fastIsing.Ising(\
                    **modelsettings)
    temps = np.linspace(0, graph.number_of_nodes(), 100)
    samps = [m.matchMagnetization(temps, 100) for i in range(10)]

    samps = np.array(samps)
    samps = samps.mean(0)
    mag, sus = samps
    snapshots    = infcy.getSnapShots(m, \
                                  nSamples = int(1e3), steps = int(1e3), \
                                  nThreads = -1)

    IDX = np.argmin(abs(mag - .8 * mag.max()))
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

idx = np.argmax(a, axis = -1)


# %%
fig, (ax, ax1) = plt.subplots(1, 2, gridspec_kw = dict(width_ratios = [1, .44]))
tmp = set()
for gidx, maxNode in enumerate(idx):
    for jdx, nudge in enumerate(nudges):
        node = maxNode[jdx]
        ax.scatter(nudge, a[gidx, jdx, node], color = colors[node])
        
        tmp.add(m.rmapping[node])
        
elements = [plt.Line2D(*[[0],[0]], marker = '.', color = colors[m.mapping[node]], label = node) for node in tmp]
#inax = ax.twinx()# ((1.05, .5, .25, .25))
ax.set(xlabel  = 'nudge size', ylabel = "largest causal impact $D_{kl}(P' || P)$")
nx.draw(graph, ax = ax1, with_labels = 1)
ax.legend(handles = elements, title = 'node')
ax.set_xlim(0, 3)
#ax.set_xscale('log')
fig.show()

# %%
for idx, nudge in enumerate(out):
    for jdx, ni in enumerate(nudge):
        fig, tax = plt.subplots()
        [tax.plot(i, color = colors[idx]) for idx, i in enumerate(ni)]
        tax.set_title(nudges[jdx])