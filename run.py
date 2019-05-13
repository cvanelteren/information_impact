#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# __author__ = 'Casper van Elteren'
"""
Created on Mon Jun 11 09:06:57 2018

@author: casper
"""


from matplotlib.pyplot import *
from numpy import *
from tqdm import tqdm
from functools import partial
from Models import fastIsing
from Toolbox import infcy
from Utils import IO, plotting as plotz
from Utils.IO import SimulationResult
from multiprocessing import cpu_count
import networkx as nx, \
        itertools, scipy,\
        os,     pickle, \
        h5py,   sys, \
        multiprocessing as mp, json,\
        datetime, sys, \
        scipy, msgpack, \
        time
close('all')

if __name__ == '__main__':
    repeats       = int(3e4)
    deltas        = 100
    step          = int(1e4)
    nSamples      = int(1e4)
    burninSamples = 0
    pulseSizes    = [1] #, -np.inf]# , .8, .7]

    nTrials       = 1
    magSide       = ''
    updateType    = 'async'
    ratios        = [.8] # , .5, .2] # if real else [.9]  # match magnetiztion at 80 percent of max

    tempres       = 50

    graphs = []
    N  = 20
    graphs = [nx.path_graph(3)]

#    graphs[0].add_edge(0,0)
#    for j in np.int32(np.logspace(0, np.log10(N-1),  5)):
#       graphs.append(nx.barabasi_albert_graph(N, j))
    if 'fs4' in os.uname().nodename or 'node' in os.uname().nodename:
        rootDirectory = '/var/scratch/cveltere/' # data storage
    else:
        rootDirectory = f'{os.getcwd()}/Data/'
# #    real = 1
# #        graphs += [nx.barabasi_albert_graph(n, i) for i in linspace(2, n - 1, 3, dtype = int)]
#     dataDir = 'Psycho' # relative path careful
#     df    = IO.readCSV(f'{dataDir}/Graph_min1_1.csv', header = 0, index_col = 0)
#     h     = IO.readCSV(f'{dataDir}/External_min1_1.csv', header = 0, index_col = 0)
#     graph   = nx.from_pandas_adjacency(df)
#     attr = {}
#     for node, row in h.iterrows():
#         attr[node] = dict(H = row['externalField'], nudges = 0)
#     nx.set_node_attributes(graph, attr)
#     graphs.append(graph)

    start = datetime.datetime.now()
    targetDirectory = rootDirectory + f'{start.isoformat()}' # make default path
    for graph in graphs:
        now =  datetime.datetime.now().isoformat()
        # if multiple graphs are tested; group them together
        if len(graphs) > 1:
            if not os.path.exists(rootDirectory):
                os.mkdir(rootDirectory)
            targetDirectory = rootDirectory + f'/{now}'
        os.mkdir(targetDirectory)


        # graph = nx.barabasi_albert_graph(10, 3)
        modelSettings = dict(\
                             graph       = graph,\
                             temperature = 0,\
                             updateType  = updateType,\
                             magSide     = magSide)
        model = fastIsing.Ising(**modelSettings)

        magRange = np.asarray(ratios)

        # magRange = array([.9, .2])
        fitTemps = linspace(0, np.log(graph.number_of_nodes()), tempres)
        mag, sus = model.matchMagnetization(temps = fitTemps,\
                                n = int(1e3), burninSamples = 0)

        func = lambda x, a, b, c, d :  a / (1 + exp(b * (x - c))) + d # tanh(-a * x)* b + c
        from lowess import lowess

        yest = lowess(fitTemps, mag)
        fig, ax = subplots()
        ax.scatter(fitTemps, yest, color = 'red')
        ax.scatter(fitTemps, mag, color = 'blue')

        # ax.scatter(fitTemps, fmag, alpha = .2)
        setp(ax, **dict(xlabel = 'Temperature', ylabel = '<M>'))
        fig.savefig(f'{targetDirectory}/temp vs mag.png')
        show()
        assert 0

        # TODO: combine these? > don't as it is model specific imo
        tmp = dict(\
                   fitTemps     = fitTemps, \
                   matchedTemps = matchedTemps, \
                   magRange     = magRange, \
                   mag          = mag,\
                   )
        settings = dict(
                    repeats          = repeats,\
                    deltas           = deltas,\
                    nSamples         = nSamples,\
                    step             = step,\
                    burninSamples    = burninSamples,\
                    pulseSizes       = pulseSizes,\
                    updateType       = updateType,\
                    nNodes           = graph.number_of_nodes(),\
                    nTrials          = nTrials,\
                    graph            = nx.node_link_data(graph),\
                    mapping          = model.mapping,\
                    rmapping         = model.rmapping,\
                    model            = type(model).__name__,\
                    directory        = targetDirectory,\
                    )
        settingsObject = IO.Settings(settings)
        settingsObject.save(targetDirectory)
        IO.savePickle(f'{targetDirectory}/mags.pickle', tmp)

        datadir = f'{targetDirectory}/Data'
        if not os.path.exists(datadir):
            os.mkdir(datadir)
        for t, mag in zip(matchedTemps, magRange):
            model.t = t # update beta
            for trial in range(nTrials):
                # st = [random.choice(model.agentStates, size = model.nNodes) for i in range(nSamples)]
                print(f'{datetime.datetime.now().isoformat()} Getting snapshots')
                # enforce no external influence
                pulse        = {}
                model.nudges = pulse
                snapshots    = infcy.getSnapShots(model, nSamples, \
                                               burninSamples = burninSamples, \
                                               steps         = step)
                # TODO: uggly, against DRY
                # always perform control
                conditional, px, mi = infcy.runMC(model, snapshots, deltas, repeats)
                # snapshots, conditional, mi = infcy.reverseCalculation(nSamples, model, deltas, pulse)[-3:]
                now = datetime.datetime.now().isoformat()
                fileName = f'{datadir}/{now}_T={t}_pulse={pulse}_trial={trial}'
                sr       = SimulationResult(\
                                        mi          = mi,\
                                        conditional = conditional,\
                                        px          = px,\
                                        snapshots   = snapshots)
                IO.savePickle(fileName, sr)

                from Utils.stats import KL
                for pulseSize in pulseSizes:
                    for n in model.graph.nodes():
                        pulse        = {n : pulseSize}
                        model.nudges = pulse
                        conditional, px, mi = infcy.runMC(model, snapshots, deltas, repeats)
                        # snapshots, conditional, mi = infcy.reverseCalculation(nSamples, model, deltas, pulse)[-3:]
                        now = datetime.datetime.now().isoformat()
                        fileName = f'{datadir}/{now}_T={t}_pulse={pulse}_trial={trial}'
                        sr       = SimulationResult(\
                                                mi          = mi,\
                                                conditional = conditional,\
                                                px          = px,\
                                                snapshots   = snapshots)
                        IO.savePickle(fileName, sr)
