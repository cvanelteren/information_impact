# -*- coding: utf-8 -*-
"""
Created on Thu May 17 10:53:00 2018

@author: Cas
"""

from matplotlib.pyplot import *
from numpy import *
import information, plotting as plotz, os, pickle, IO, fastIsing, networkx as nx
from tqdm import tqdm

if __name__ == '__main__':
    graph = nx.path_graph(3, nx.DiGraph())
#    graph.add_edge(0, 2
#    )
#    graph = nx.DiGraph();
#    graph.add_edge(0, 2)
#    graph.add_edge(1, 2)
    
    
    temperatures = logspace(-10, 0, 20)
    temperatures = [.001]
    nSamples = 1000
    step     = 1
    deltas   = 10
    repeats = 1000
    res = {}
    for temperature in tqdm(temperatures):
        model = fastIsing.Ising(graph, temperature, False)
        snapshots = information.getSnapShots(model, nSamples, step)[0]
        mc = information.monteCarlo(model, snapshots, deltas, repeats)
        pState, pNode, pNodeGivenState, joint, I = information.mutualInformationShift(model, snapshots, mc,
                                                                                      mode = 'sinc')
        
        res[temperature] = dict(state = pState, node = pNode, pxy = pNodeGivenState, \
           joint = joint, I = I, mc = mc)
        
# %%
#    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
#    close('all')
#    s  = 17
#    fig, ax = subplots(2, sharex = 'all');
#    for key, value in res.items():
#        p = value['pxy'].squeeze()
#        t = nanmean(plotz.hellingerDistance(p[...,[0]], p[...,[1]]), -1)
#        
#        [ax[0].plot(i, '.', color = cm.tab10(idx), alpha = key, markersize = s) for idx, i in enumerate(value["I"].T)]
#        [ax[1].plot(i, '.', color = cm.tab10(idx), alpha = clip(key, .1, 1), markersize = s) for idx, i in enumerate(t.T)]
##        ax[1].plot(t[:, :], '.', color = cm.tab20(key))
#        
#    axx = inset_axes(ax[0], width = '40%', height = '60%')
#    plotz.addGraphPretty(model, axx, cmap = cm.tab10)
#    axx.axis('off')
#    axx.set_aspect('equal')
#    ax[0].set_ylabel('Source MI')
#    ax[1].set_ylabel('$<H>_X$')
#    ax[1].set_xlabel('Time [step]')
#    
#    # %%
#    fig, ax = subplots();
#    for key, value in res.items():
#        s = value['state']
#        s = s.reshape(-1, s.shape[-1])
#        ss = information.entropy(s.mean(0))
#        ax.scatter(key, ss, color = 'blue')
#    ax.set_xlabel('Temperature') ; ax.set_ylabel('State entropy')
    # %%
    aa = {(-1, -1, 1), (-1, -1 ,-1), (-1, 1, 1), (-1, 1, -1)}
    test = information.monteCarlo(model, snapshots = aa, deltas = 10, repeats = 1000)
    # %%
    test = mc
    p = zeros((deltas + 1, model.nNodes, 2, 8))
    for idx, state in enumerate(test):
        for sample in state:
            for tidx, t_sample in enumerate(sample):
                for n, i in enumerate(t_sample):
                    if sample[0, n] == 1 and i == 1:
                        p[tidx, n, 0, idx] += 1
                        
                    elif sample[0, n] == 1 and i == -1:
                        p[tidx, n, 1, idx] += 1
                        
                    elif sample[0, n] == -1 and i == 1:
                        p[tidx, n, 0, idx] += 1
                        
                    elif sample[0, n] == -1 and i == -1:
                        p[tidx, n, 1, idx] += 1
    pp = p.sum(-1) / (test.shape[0] * test.shape[1])
    dp = p/repeats
    
    h = plotz.hellingerDistance(pp[..., [0]], pp[..., [1]])
    hh = plotz.hellingerDistance(dp[:, :, 0, :], dp[:, :, 1, :])
    hhh = plotz.hellingerDistance(pNodeGivenState[..., [0]], pNodeGivenState[..., [1]]).mean(-1)
    fig, ax = subplots(); ax.plot(h)
    fig, ax = subplots(); ax.plot(hh, '.')
    fig, ax = subplots(); ax.plot(hhh)