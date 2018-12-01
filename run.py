#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 11 09:06:57 2018

@author: casper
"""

import fastIsing, networkx as nx, itertools, plotting as plotz, RBN
from sklearn.gaussian_process import GaussianProcessRegressor as gp
import infcy, information, scipy
from matplotlib.pyplot import *
from numpy import *
from tqdm import tqdm
from functools import partial
from multiprocessing import Pool
from scipy import sparse
import os, pickle, IO, h5py, sys
import IO, multiprocessing as mp, json
import datetime
from artNet import Net
from time import time
close('all')
np.random.seed() # set seed
if __name__ == '__main__':
    real          = False
    repeats       = int(1e4) if real else 1000
    deltas        = 10       if real else 10
    step          = 1
    nSamples      = int(1e4) if real else 1000
    burninSamples = 100
    pulseSize     = 1

    numIter       = 25 if real else 5
    magSide       = 'neg'
    updateMethod  = 'async'
    CHECK         = [.9, .8, .7]  if real else [.9]  # match magnetiztion at 80 percent of max

#     dataDir = 'Psycho' # relative path careful
#     df    = IO.readCSV(f'{dataDir}/Graph_min1_1.csv', header = 0, index_col = 0)
#     h     = IO.readCSV(f'{dataDir}/External_min1_1.csv', header = 0, index_col = 0)
# #
#     graph   = nx.from_pandas_adjacency(df)
#     for i, j in graph.edges():
#         graph[i][j]['weight'] = 1
#     #
#     attr = {}
#     for node, row in h.iterrows():
#         attr[node] = dict(H = row['externalField'], nudges = 0)
#     nx.set_node_attributes(graph, attr)

    n = 10
    x = linspace(1, n, 3, dtype = int)
    # graphs = [nx.barabasi_albert_graph(n, i) \
    #           for i in x]
    graphs = [nx.barabasi_albert_graph(10, 2)]
    for graph in graphs:
        now = time()
        targetDirectory = f'{os.getcwd()}/Data/{now}'
        os.mkdir(targetDirectory)
        settings = dict(
            repeat           = repeats,
            deltas           = deltas,
            nSamples         = nSamples,
            step             = step,
            burninSamples    = burninSamples,
            pulseSize        = pulseSize,
            updateMethod     = updateMethod
                          )
        IO.saveSettings(targetDirectory, settings)

        # graph = nx.barabasi_albert_graph(10, 3)
        model = fastIsing.Ising(graph = graph, \
                                temperature = 0, \
                                mode = updateMethod, magSide = magSide)


        conditions = {(tuple(model.nodeIDs), (i,)) : \
        idx for idx, i in enumerate(model.nodeIDs)}
        # %%
    #    f = 'nSamples=10000_k=10_deltas=5_modesource_t=10_n=65.h5'
    #    fileName = f'Data/{f}'
        mode = model.mode
        # match the temperature to sample from
        # magRange = [.2]
        if os.path.isfile(f'{targetDirectory}/mags.pickle'):
            tmp = IO.loadPickle(f'{targetDirectory}/mags.pickle')
            for i, j in tmp.items():
                globals()[i] = j
        else:
            magRange = array([CHECK]) if isinstance(CHECK, float) else array(CHECK)

            # magRange = array([.9, .2])
            temps = linspace(.1, 10, 100)

            mag, sus = model.matchMagnetization(  temps = temps,\
             n = 1000, burninSamples = 0)


            func = lambda x, a, b, c, d :  a / (1 + exp(b * (x - c))) + d # tanh(-a * x)* b + c
            # func = lambda x, a, b, c : a + b*exp(-c * x)
            a, b = scipy.optimize.curve_fit(func, temps, mag.squeeze(), maxfev = 10000)

            # run the simulation per temperature
            temperatures = array([])
            f_root = lambda x,  c: func(x, *a) - c
            magRange *= max(mag)
            for m in magRange:
                r = scipy.optimize.root(f_root, 0, args = (m), method = 'linearmixing')#, method = 'linearmixing')
                rot = r.x if r.x > 0 else 0
                temperatures = hstack((temperatures, rot))

            fig, ax = subplots()
            xx = linspace(0, max(temps), 1000)
            ax.plot(xx, func(xx, *a))
            ax.scatter(temperatures, func(temperatures, *a), c ='red')
            ax.scatter(temps, mag, alpha = .2)
            setp(ax, **dict(xlabel = 'Temperature', ylabel = '<M>'))
            savefig(f'{targetDirectory}/temp vs mag.png')
            # show()
            tmp = dict(temps = temps, \
            temperatures = temperatures, magRange = magRange, mag = mag)
            IO.savePickle(f'{targetDirectory}/mags.pickle', tmp)


        for t in temperatures:
            baseLength = 20
            s = f'Setting T={t:.2f}:'
            print(f'{s}{" " * (baseLength-len(s))}\t{datetime.datetime.now()}')
            model.t = t
            model.reset()
            for i in range(numIter):
                from multiprocessing import cpu_count
                # st = [random.choice(model.agentStates, size = model.nNodes) for i in range(nSamples)]
                s = f'Getting snapshots:'
                print(f'{s}{" " * (baseLength-len(s))}\t{datetime.datetime.now()}')
                # snapshots = {tuple(infcy.encodeState(i))}
                pulse = {}
                model.pulse = pulse
                snapshots   = infcy.getSnapShots(model, nSamples, \
                                               parallel      = cpu_count(), \
                                               burninSamples = burninSamples, \
                                               step          = step)
                s = "Starting MC:"
                print(f'{s}{" " * (baseLength-len(s))}\t{datetime.datetime.now()}')
                conditional = infcy.monteCarlo_alt(\
                                               model  = model, snapshots = snapshots,\
                                               deltas = deltas, repeats  = repeats,\
                                               mode   = mode)

                # px, conditional, snapshots, mi = infcy.reverseCalculation(nSamples, model, deltas, pulse)[-4:]
                # conditional = infcy.monteCarlo(model = model, snapshots = snapshots, conditions = conditions,\
                 # deltas = deltas, repeats = repeats, pulse = pulse, mode = 'source')

                s = f'Computing MI:'
                print(f'{s}{" " * (baseLength-len(s))}\t{datetime.datetime.now()}')
                px, mi = infcy.mutualInformation_alt(\
                conditional, deltas, snapshots, model)
                # mi   = array([infcy.mutualInformation(joint, condition, deltas) for condition in conditions.values()])

                fileName = f'{targetDirectory}/{time()}_nSamples={nSamples}_k={repeats}_deltas={deltas}_mode_{mode}_t={t}_n={model.nNodes}_pulse={pulse}.pickle'
                sr = IO.SimulationResult(mi = mi,
                                        conditional = conditional,
                                        model = model,\
                                        px = px, snapshots = snapshots)
                IO.savePickle(fileName, sr)
                pulses = {node : pulseSize for node in model.graph.nodes()}
                for n, p in pulses.items():
                    pulse = {n : p}
                    model.pulse = pulse
                    ss = time()
                    conditional = infcy.monteCarlo_alt(model = model, snapshots = snapshots,\
                                            deltas = deltas, repeats = repeats,\
                                            mode = mode)
                    print(time() - ss)
                    s = f'Computing MI:'
                    print(f'{s}{" " * (baseLength-len(s))}\t{datetime.datetime.now()}')
                    px, mi = infcy.mutualInformation_alt(conditional, deltas, snapshots, model)
                    # snapshots, conditional, mi = infcy.reverseCalculation(nSamples, model, deltas, pulse)[-3:]
                    fileName = f'{targetDirectory}/{time()}_nSamples={nSamples}_k={repeats}_deltas={deltas}_mode_{mode}_t={t}_n={model.nNodes}_pulse={pulse}.pickle'
                    sr = IO.SimulationResult(mi = mi,
                                            conditional = conditional,
                                            model = model,\
                                            px = px, snapshots = snapshots)
                    IO.savePickle(fileName, sr)
