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
repeats = int(1e4)
start = time.time()

N = 5 # repeats?
#nudges[-1] = np.inf
#graph = nx.barabasi_albert_graph(10, 2)
graph = nx.erdos_renyi_graph(5, .2)
#graph = nx.read_weighted_edgelist('Graphs/aves-barn-swallow-non-physical.edges')
# %%
#graph = nx.krackhardt_kite_graph()


dataDir = 'Graphs' # relative path careful
df    = IO.readCSV(f'{dataDir}/Graph_min1_1.csv', header = 0, index_col = 0)
h     = IO.readCSV(f'{dataDir}/External_min1_1.csv', header = 0, index_col = 0)
#graph   = nx.from_pandas_adjacency(df)
#graph = nx.erdos_renyi_graph(10, .3)
#graph = nx.path_graph(5)
#graph = nx.Graph()
#graph.add_edge(0,1)
#graph.add_edge(1,2)
#graph.add_edge(2,0)
#graph[0][1]['weight'] = 10
#attr = {}
graph = nx.krackhardt_kite_graph()

#for node, row in h.iterrows():
#    attr[node] = dict(H = row['externalField'], nudges = 0)
from scipy.ndimage import gaussian_filter1d

nudges = np.logspace(-1, 1, 10)
out = np.zeros((N, nudges.size, deltas))
mis = np.zeros((N, graph.number_of_nodes(), deltas))

fig, ax = plt.subplots()
tax = ax.twinx()


modelsettings = dict(\
                 graph = graph, \
                 updateType = 'single', \
                 nudgeType  = 'constant',\
                 magSide = 'neg')
m = fastIsing.Ising(\
                **modelsettings)
temps = np.logspace(-6, np.log10(20), 30)
samps = [m.matchMagnetization(temps, 100) for i in range(1)]

samps = np.array(samps)
samps = gaussian_filter1d(samps, 3, axis = -1)
samps = samps.mean(0)
mag, sus = samps
ax.plot(temps, mag)
tax.plot(temps, sus)

fig.show()
IDX = np.argmin(abs(mag - .8 * mag.max()))
#    IDX = np.argmax(sus)
m.t = temps[IDX]
mis[n] = mi.T

storedTemperatures = []
for n in range(N):
    snapshots    = infcy.getSnapShots(m, \
                                  nSamples = int(1e4), steps = int(1e3), \
                                  nThreads = -1)


#    storedTemperatures.append(temps[IDX])
    conditional, px, mi = infcy.runMC(m, snapshots, deltas, repeats)

    auc = np.trapz(mi[deltas // 2 :, ], axis = 0)
    idx = np.argmax(auc)
    node = m.rmapping[idx]
    
    for nidx, nudge in enumerate(nudges):
        tmp = {node : nudge}
        m.nudges = tmp
        c, p, n_ = infcy.runMC(m, snapshots, deltas, repeats)
        out[n, nidx, :] = JS(px, p).sum(-1)
#        for node, idx in m.mapping.items():
#            m.nudges = {node : nudge}
#            c, p, n_ = infcy.runMC(m, snapshots, deltas, repeats)
#            out[n, nidx, idx, :] = KL(px, p).sum(-1)
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

mu = b.mean(0)
kap = np.median(b, axis = 0)
sig = b.std(0) * 2 
fig, ax = plt.subplots()
ax.errorbar(nudges, mu, yerr = sig, capsize = 10, label = r'mean $\pm$ 2 * $\sigma$')

[ax.scatter(nudges, i, color = 'k') for i in b]
ax.plot(nudges, kap, color = 'red', label = 'median')


grad = abs(np.gradient(kap, nudges, edge_order = 2))
vargrad = np.gradient(sig, nudges)
idx = np.argmax(grad)
jdx = np.argmin(vargrad)


optimizedNudge = nudges[idx]
varOpt = nudges[jdx]
ax.axvline(optimizedNudge, color = 'green', linestyle = 'dashed')
ax.axvline(varOpt, color = 'purple', linestyle = 'dashed')
ax.plot(nudges, grad, label = 'median gradient')
ax.plot(nudges, vargrad, label = 'STD gradient')
#ax.set_yscale('log')
#ax.set_xscale('log')
ax.set_ylabel('Causal impact')
ax.set_xlabel('Nudge size')
ax.legend()
#ax.set_yscale('log')

# %%

nudgedData = np.zeros((m.nNodes, deltas))
m.t = np.mean(storedTemperatures)
for k, v in m.mapping.items():
    nudge = {k : optimizedNudge}
    m.nudges = nudge
    c, p, n_ = infcy.runMC(m, snapshots, deltas, repeats)
    nudgedData[v] = KL(px, p).sum(-1)
# %%
fig, (ax1, ax2) = plt.subplots(1, 2, sharex = 'all')
colors = plt.cm.tab20(np.arange(m.nNodes))
mi = mis.mean(0)
for node, idx in m.mapping.items():
    ax1.plot(nudgedData[idx, deltas // 2:], linestyle = 'dashed', label = node, color = colors[idx])
    ax2.plot(mi[idx], color = colors[idx], label = node)
ax2.legend()
ax1.set(xlim = (0, 30))

ax1.set(ylabel = 'causal')
ax2.set(ylabel = 'information')
fig.subplots_adjust(wspace = .3)
#ax.set(xlim = (0, 10))
fig.show()
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