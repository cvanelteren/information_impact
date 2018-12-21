#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# __author__ = 'Casper van Elteren'
"""
Created on Mon Jun 11 09:06:57 2018

@author: casper
"""

from Models import fastIsing
from Toolbox import infcy
from Utils import IO, plotting as plotz
from Utils.IO import SimulationResult
import networkx as nx, itertools, scipy,\
        os, pickle, h5py, sys, multiprocessing as mp, json,\
        datetime, sys
import time
from matplotlib.pyplot import *
from numpy import *
from tqdm import tqdm
from functools import partial
from scipy import sparse
close('all')
if __name__ == '__main__':
    if len(sys.argv) > 1:
        real = sys.argv[1]
    else:
        real = 0
    repeats       = int(1e4)
    deltas        = 20
    step          = 100
    nSamples      = int(1e4)
    burninSamples = 100_000
    pulseSize     = .5
    numIter       = 5 if real else 5
    magSide       = 'neg'
    updateType    = 'single'
    CHECK         = [.9] # [.9, .8, .7]  if real else [.9]  # match magnetiztion at 80 percent of max
    n = 10
    graphs = []
#    real = 1
    if real:
#        graphs += [nx.barabasi_albert_graph(n, i) for i in linspace(2, n - 1, 3, dtype = int)]
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
        graphs.append(graph)
    else:
#       graphs += [nx.path_graph(3)]
        graphs += [nx.krackhardt_kite_graph()]


    # graphs = [nx.barabasi_albert_graph(10,5)]
#    graphs = [nx.path_graph(3)]

    for graph in graphs:
        now = time.time()
        targetDirectory = f'{os.getcwd()}/Data/{now}'
        os.mkdir(targetDirectory)
        settings = dict(
            repeat           = repeats,
            deltas           = deltas,
            nSamples         = nSamples,
            step             = step,
            burninSamples    = burninSamples,
            pulseSize        = pulseSize,
            updateMethod     = updateType
                          )
        IO.saveSettings(targetDirectory, settings)

        # graph = nx.barabasi_albert_graph(10, 3)
        modelSettings = dict(\
                             graph       = graph,\
                             temperature = 0,\
                             updateType  = updateType,\
                             magSide     = magSide)
        model = fastIsing.Ising(**modelSettings)
#        print(model.mapping.items())
#        assert 0

    #    f = 'nSamples=10000_k=10_deltas=5_modesource_t=10_n=65.h5'
    #    fileName = f'Data/{f}'
        updateType = model.updateType
        # match the temperature to sample from
        # magRange = [.2]
        if os.path.isfile(f'{targetDirectory}/mags.pickle'):
            tmp = IO.loadPickle(f'{targetDirectory}/mags.pickle')
            for i, j in tmp.items():
                globals()[i] = j
        else:
            magRange = array([CHECK]) if isinstance(CHECK, float) else array(CHECK)

            # magRange = array([.9, .2])
            temps = linspace(0, 10, 100)

#            model.magSide = 'neg'
            mag, sus = model.matchMagnetization(temps = temps,\
             n = 10000)


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
            print(f'{time.time()} Setting {t}')
            model.t = t # update beta
            for i in range(numIter):
                from multiprocessing import cpu_count
                # st = [random.choice(model.agentStates, size = model.nNodes) for i in range(nSamples)]
                print(f'{time.time()} Getting snapshots')
                pulse        = {}
                model.nudges = pulse
                snapshots    = infcy.getSnapShots(model, nSamples, \
                                               burninSamples = burninSamples, \
                                               step          = step)



                print(f'{time.time()}')
                conditional = infcy.monteCarlo(\
                                               model  = model, snapshots = snapshots,\
                                               deltas = deltas, repeats  = repeats,\
                                               )


                # px, conditional, snapshots, mi = infcy.reverseCalculation(nSamples, model, deltas, pulse)[-4:]
                # conditional = infcy.monteCarlo(model = model, snapshots = snapshots, conditions = conditions,\
                 # deltas = deltas, repeats = repeats, pulse = pulse, updateType= 'source')

                print(f'{time.time()} Computing MI')
                px, mi = infcy.mutualInformation(\
                conditional, deltas, snapshots, model)
                # mi   = array([infcy.mutualInformation(joint, condition, deltas) for condition in conditions.values()])
                fileName = f'{targetDirectory}/{time.time()}_nSamples={nSamples}_k={repeats}_deltas={deltas}_mode_{updateType}_t={t}_n={model.nNodes}_pulse={pulse}.pickle'
                sr = SimulationResult(\
                                         mi         = mi,\
                                        conditional = conditional,\
                                        graph       = model.graph,\
                                        px          = px,\
                                        snapshots   = snapshots)
                IO.savePickle(fileName, sr)

                # estimate average energy
                # pulses = {}
                # for k, v in snapshots.items():
                #     state = infcy.decodeState(k, model.nNodes)
                #     for i in range(model.nNodes):
                #         nodei = model.rmapping[i]
                #         e = 0
                #         for nodej in model.graph.neighbors(nodei):
                #             j = model.mapping[nodej]
                #             e += state[j] * state[i] * model.graph[nodei][nodej]['weight']
                #         pulses[nodei] = pulses.get(nodei, 0)  - e * v - state[i] * model.H[i]
                # for k in pulses:
                #     pulses[k] *= pulseSize

#
#
#                # nudge all nodes
                pulses = {node : pulseSize for node in model.graph.nodes()}
                for n, p in pulses.items():
                    pulse        = {n : p}
                    model.nudges = pulse
                    conditional = infcy.monteCarlo(model = model, snapshots = snapshots,\
                                            deltas = deltas, repeats = repeats,\
                                            )
                    print(f'{time.time()} Computing MI')
                    px, mi = infcy.mutualInformation(conditional, deltas, snapshots, model)
                    # snapshots, conditional, mi = infcy.reverseCalculation(nSamples, model, deltas, pulse)[-3:]
                    fileName = f'{targetDirectory}/{time.time()}_nSamples={nSamples}_k={repeats}_deltas={deltas}_mode_{updateType}_t={t}_n={model.nNodes}_pulse={pulse}.pickle'
                    sr = SimulationResult(\
                                            mi          = mi,\
                                            conditional = conditional,\
                                            graph       = model.graph,\
                                            px          = px,\
                                            snapshots   = snapshots)
                    IO.savePickle(fileName, sr)
