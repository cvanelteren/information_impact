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

from matplotlib.pyplot import *
from numpy import *
from tqdm import tqdm
from functools import partial

import (networkx as nx, \
        itertools, scipy,\
        os,     pickle, \
        h5py,   sys, \
        multiprocessing as mp, json,\
        datetime, sys, \
        scipy, msgpack, \
        time)
close('all')
if __name__ == '__main__':
    if len(sys.argv) > 1:
        real = sys.argv[1]
    else:
        real = 0
    repeats       = int(1e4)
    deltas        = 20
    step          = int(1e4)
    nSamples      = int(1e2)
    burninSamples = 0
    pulseSizes    = [1]#, inf] #, -np.inf]# , .8, .7]

    nTrials       = 10
    magSide       = ''
    updateType    = 'single'
    CHECK         = [.9] # , .5, .2] # if real else [.9]  # match magnetiztion at 80 percent of max

    tempres       = 100
    graphs = []
#    real = 1
    if real:
#        graphs += [nx.barabasi_albert_graph(n, i) for i in linspace(2, n - 1, 3, dtype = int)]
        dataDir = 'Psycho' # relative path careful
        df    = IO.readCSV(f'{dataDir}/Graph_min1_1.csv', header = 0, index_col = 0)
        h     = IO.readCSV(f'{dataDir}/External_min1_1.csv', header = 0, index_col = 0)
        graph   = nx.from_pandas_adjacency(df)
        attr = {}
        for node, row in h.iterrows():
            attr[node] = dict(H = row['externalField'], nudges = 0)
        nx.set_node_attributes(graph, attr)
        graphs.append(graph)
    else:
       N = 50
       tmp = logspace(0, log10(N - 1), 10, dtype = int)
       graphs += [nx.barabasi_albert_graph(N, ni) for ni in tmp]
       # graphs += [nx.krackhardt_kite_graph()]
       graphs = [nx.path_graph(10)]
       # n = 10
       # nn = 10
       # for i in [3, 5, n - 1]:
           # for a in range(nn):
               # graphs.append(nx.barabasi_albert_graph(n, i))
           # graphs += [nx.barabasi_albert_graph(n, i) for i in range(1, n)]


    # graphs = [nx.barabasi_albert_graph(10,5)]
#    graphs = [nx.path_graph(3)]

    rootDirectory = f'{os.getcwd()}/Data/{time.time()}'  if len(graphs) > 1 else ''
    for graph in graphs:
        now = time.time()

        # group graphs together if this is run
        if rootDirectory == '':
            targetDirectory = f'{os.getcwd()}/Data/{now}'
        # group graphs; setup paths
        else:
            if not os.path.exists(rootDirectory):
                os.mkdir(rootDirectory)
            targetDirectory = rootDirectory + f'/{now}'

        os.mkdir(targetDirectory)
        settings = dict(
            repeat           = repeats,
            deltas           = deltas,
            nSamples         = nSamples,
            step             = step,
            burninSamples    = burninSamples,
            pulseSizes       = pulseSizes,
            updateMethod     = updateType,\
            nNodes           = graph.number_of_nodes(),
            nTrials         = nTrials,\
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
            temps = linspace(0, 15, tempres)
            mag, sus = model.matchMagnetization(temps = temps,\
             n = int(1e4), burninSamples = 0)


            func = lambda x, a, b, c, d :  a / (1 + exp(b * (x - c))) + d # tanh(-a * x)* b + c
            # func = lambda x, a, b, c : a + b*exp(-c * x)
            fmag = scipy.ndimage.gaussian_filter1d(mag, 2)
            a, b = scipy.optimize.curve_fit(func, temps, fmag.squeeze(), maxfev = 10000)

            # run the simulation per temperature
            temperatures = array([])
            f_root = lambda x,  c: func(x, *a) - c
            magnetizations = max(fmag) * magRange
            for m in magnetizations:
                r = scipy.optimize.root(f_root, 0, args = (m), method = 'linearmixing')#, method = 'linearmixing')
                rot = r.x if r.x > 0 else 0
                temperatures = hstack((temperatures, rot))

            fig, ax = subplots()
            xx = linspace(0, max(temps), 1000)
            ax.plot(xx, func(xx, *a))
            ax.scatter(temperatures, func(temperatures, *a), c ='red')
            ax.scatter(temps, mag, alpha = .2)
            ax.scatter(temps, fmag, alpha = .2)
            setp(ax, **dict(xlabel = 'Temperature', ylabel = '<M>'))
            savefig(f'{targetDirectory}/temp vs mag.png')
            # show()
            tmp = dict(\
                       temps        = temps, \
                       temperatures = temperatures, \
                       magRange     = magRange, \
                       mag          = mag,\
                       fmag         = fmag)
            IO.savePickle(f'{targetDirectory}/mags.pickle', tmp)

        for t, mag in zip(temperatures, magRange):
            print(f'{time.time()} Setting {t}')
            model.t = t # update beta
            tempDir = f'{targetDirectory}/{mag}'
            if not os.path.exists(tempDir):
                print('making directory')
                os.mkdir(tempDir)

            for trial in range(nTrials):
                from multiprocessing import cpu_count
                # st = [random.choice(model.agentStates, size = model.nNodes) for i in range(nSamples)]
                print(f'{time.time()} Getting snapshots')
                # enforce no external influence
                pulse        = {}
                model.nudges = pulse
                snapshots    = infcy.getSnapShots(model, nSamples, \
                                               burninSamples = burninSamples, \
                                               steps         = step)
                # TODO: uggly, against DRY
                # always perform control
                conditional, px, mi = infcy.runMC(model, snapshots, deltas, repeats)
                print(f'{time.time()} Computing MI')
                # snapshots, conditional, mi = infcy.reverseCalculation(nSamples, model, deltas, pulse)[-3:]
                if not os.path.exists(f'{tempDir}/control/'):
                    os.mkdir(f'{tempDir}/control')
                fileName = f'{tempDir}/control/{time.time()}_nSamples ={nSamples}_k ={repeats}_deltas ={deltas}_mode_{updateType}_t={t}_n ={model.nNodes}_pulse ={pulse}.pickle'
                sr       = SimulationResult(\
                                        mi          = mi,\
                                        conditional = conditional,\
                                        graph       = model.graph,\
                                        px          = px,\
                                        snapshots   = snapshots)
                IO.savePickle(fileName, sr)
                for pulseSize in pulseSizes:
                    pulseDir = f'{tempDir}/{pulseSize}'
                    if not os.path.exists(pulseDir):
                        os.mkdir(pulseDir)
                    for n in model.graph.nodes():
                        pulse        = {n : pulseSize}
                        model.nudges = pulse
                        conditional, px, mi = infcy.runMC(model, snapshots, deltas, repeats)

                        print(f'{time.time()} Computing MI')
                        # snapshots, conditional, mi = infcy.reverseCalculation(nSamples, model, deltas, pulse)[-3:]
                        fileName = f'{pulseDir}/{time.time()}_nSamples ={nSamples}_k ={repeats}_deltas ={deltas}_mode_{updateType}_t={t}_n ={model.nNodes}_pulse ={pulse}.pickle'
                        sr       = SimulationResult(\
                                                mi          = mi,\
                                                conditional = conditional,\
                                                graph       = model.graph,\
                                                px          = px,\
                                                snapshots   = snapshots)
                        IO.savePickle(fileName, sr)

                # estimate average energy
                #     for i in range(model.nNodes):
                #         nodei = model.rmapping[i]
                #         e = 0
                #         for nodej in model.graph.neighbors(nodei):
                #             j = model.mapping[nodej]
                #             e += state[j] * state[i] * model.graph[nodei][nodej]['weight']
                #         pulses[nodei] = pulses.get(nodei, 0)  + e * v + state[i] * model.H[i]
                # for k in pulses:
                #     pulses[k] *= pulseSize
