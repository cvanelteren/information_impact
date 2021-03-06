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


from Models import FastIsing
from Toolbox import infcy
from Utils import IO, plotting as plotz
from Utils.IO import SimulationResult

import networkx as nx, \
        itertools, scipy,\
        os,     pickle, \
        sys, \
        multiprocessing as mp, json,\
        datetime, sys, \
        scipy, \
        time
close('all')
repeats       = int(1e5)
deltas        = 30
step          = int(1e3)
nSamples      = int(1e4)
burninSamples = 0
pulseSizes    = np.linspace(.1, 1, 9).tolist()
# pulseSizes    = np.arange(0.5, 5, .5) # np.linspace(0.5, 5, 5).tolist() # [0.5,  np.inf] #, -np.inf]# , .8, .7]

nTrials       = 20
magSide       = ''
updateType    = '0.25'
CHECK         = [0.8] # , .5, .2] # if real else [.9]  # match magnetiztion at 80 percent of max
nudgeType     = 'constant'
tempres       = 10 #100

loadGraph = ''
if __name__ == '__main__':
    graphs = []
    N  = 10
    if not loadGraph:
        for i in range(10):

            r = np.random.rand() * (1 - .2) + .2
            # g = nx.barabasi_albert_graph(N, 2)
            # g = nx.erdos_renyi_graph(N, r)
            # g = nx.duplication_divergence_graph(N, r)
            # graphs.append(g)
    else:
        print('running craph graph')
        graph = IO.loadPickle(loadGraph)
        graphs.append(graph)
       # w = nx.utils.powerlaw_sequence(N, 2)
       # g = nx.expected_degree_graph(w)
        # g = sorted(nx.connected_component_subgraphs(g), key = lambda x: len(x))[-1]

        #for i, j in g.edges():
        #    g[i][j]['weight'] = np.random.rand() * 2 - 1
#        graphs.append(g)

#    graphs[0].add_edge(0,0)
#    for j in np.int32(np.logspace(0, np.log10(N-1),  5)):
#       graphs.append(nx.barabasi_albert_graph(N, j))
    dataDir = 'Graphs' # relative path careful
    df    = IO.readCSV(f'{dataDir}/Graph_min1_1.csv', header = 0, index_col = 0)
    h     = IO.readCSV(f'{dataDir}/External_min1_1.csv', header = 0, index_col = 0)
    graph   = nx.from_pandas_adjacency(df)
    attr = {}
    for node, row in h.iterrows():
        attr[node] = dict(H = row['externalField'], nudges = 0)
    nx.set_node_attributes(graph, attr)
    graphs.append(graph)

    if 'fs4' in os.uname().nodename or 'node' in os.uname().nodename:
        now = datetime.datetime.now().isoformat()
        rootDirectory = f'/var/scratch/cveltere/{now}' # data storage
    else:
        rootDirectory = f'{os.getcwd()}/Data/'
# #    real = 1
#         graphs += [nx.barabasi_albert_graph(n, i) for i in linspace(2, n - 1, 3, dtype = int)]


    for graph in graphs:
        start = datetime.datetime.now()
        targetDirectory = os.path.join(rootDirectory, f'{start.isoformat()}')
        now = datetime.datetime.now().isoformat()
        # if multiple graphs are tested; group them together
        if not os.path.exists(rootDirectory):
            os.mkdir(rootDirectory)
        os.mkdir(targetDirectory)


        # graph = nx.barabasi_albert_graph(10, 3)
        modelSettings = dict(\
                             graph       = graph,\
                             temperature = 0,\
                             updateType  = updateType,\
                             magSide     = magSide,\
                             nudgeType   = nudgeType)
        model = FastIsing.Ising(**modelSettings)
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
            fitTemps = linspace(0, graph.number_of_nodes() / 2, tempres)
            mag, sus = model.matchMagnetization(temperatures = fitTemps,\
             n = int(1e3), burninSamples = 0)
            print('>>', mag,sus)

            func = lambda x, a, b, c, d :  a / (1 + exp(b * (x - c))) + d # tanh(-a * x)* b + c
            # func = lambda x, a, b, c : a + b*exp(-c * x)
            fmag = scipy.ndimage.gaussian_filter1d(mag, .2)
            a, b = scipy.optimize.curve_fit(func, fitTemps, fmag.squeeze(), maxfev = 10000)

            matchedTemps = array([])
            f_root = lambda x,  c: func(x, *a) - c
            magnetizations = max(fmag) * magRange
            for m in magnetizations:
                r = scipy.optimize.root(f_root, 0, args = (m), method = 'linearmixing')#, method = 'linearmixing')
                rot = r.x if r.x > 0 else 0
                matchedTemps = hstack((matchedTemps, rot))

            fig, ax = subplots()
            xx = linspace(0, max(fitTemps), 1000)
            ax.plot(xx, func(xx, *a))
            ax.scatter(matchedTemps, func(matchedTemps, *a), c ='red')
            ax.scatter(fitTemps, mag, alpha = .2)
            ax.scatter(fitTemps, fmag, alpha = .2)
            setp(ax, **dict(xlabel = 'Temperature', ylabel = '<M>'))
#            savefig(f'{targetDirectory}/temp vs mag.png')
            # show()

            # TODO: combine these? > don't as it is model specific imo
            tmp = dict(\
                       fitTemps     = fitTemps, \
                       matchedTemps = matchedTemps, \
                       magRange     = magRange, \
                       mag          = mag,\
                       fmag         = fmag,\
                       )

            settings = dict(
                        repeats       = repeats,\
                        deltas        = deltas,\
                        nSamples      = nSamples,\
                        step          = step,\
                        burninSamples = burninSamples,\
                        pulseSizes    = pulseSizes,\
                        updateType    = updateType,\
                        nNodes        = graph.number_of_nodes(),\
                        nTrials       = nTrials,\
                        # this is added
                        graph         = nx.readwrite.json_graph.node_link_data(graph),\
                        mapping       = model.mapping,\
                        rmapping      = model.rmapping,\
                        model         = type(model).__name__,\
                        directory     = targetDirectory,\
                        nudgeType     = nudgeType,\
                        )
            settingsObject = IO.Settings(settings)
            settingsObject.save(targetDirectory)
            IO.savePickle(f'{targetDirectory}/mags.pickle', tmp)

        for t, mag in zip(matchedTemps, magRange):
            print(f'{datetime.datetime.now().isoformat()} Setting {t}')
            model.t = t # update beta
            tempDir = f'{targetDirectory}/{mag}'
            if not os.path.exists(tempDir):
                print('making directory')
                os.mkdir(tempDir)

            for trial in range(nTrials):
                from multiprocessing import cpu_count
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

                # hacked in nudge optimization
                #from Utils.plotting import fit
                #from Utils.stats import aucs
                #dex = lambda x, a, b, c, d, e, f, g: a + b * np.exp(- c * (x - d)) + e * np.exp(- f*(x - g))
                #areas = aucs(mi.T, dex, params = dict(maxfev = int(1e6), bounds = (0, np.inf)))[0]
                #node = np.argmax(aucs)
                #nudge = infcy.optimizeNudge(model, node, snapshots, deltas, repeats)
                #pulseSizes = [nudge, np.inf]
                #settings['pulseSizes'] = pulseSizes
                #settingsObject = IO.Settings(settings)
                #settingsObject.save(targetDirectory)


                print(f'{datetime.datetime.now().isoformat()} Computing MI')
                # snapshots, conditional, mi = infcy.reverseCalculation(nSamples, model, deltas, pulse)[-3:]
                if not os.path.exists(f'{tempDir}/control/'):
                    os.mkdir(f'{tempDir}/control')

                props = "nSamples deltas repeats updateType".split()
                fileName = f"{tempDir}/control/{datetime.datetime.now().isoformat()}"
                fileName += "".join(f"_{key}={settings.get(key, '')}" for key in props)
                fileName += f'_pulse={pulse}'
                sr       = SimulationResult(\
                                        mi          = mi,\
                                        conditional = conditional,\
#                                        graph       = model.graph,\
                                        px          = px,\
                                        snapshots   = snapshots)
                IO.savePickle(fileName, sr)


                from Utils.stats import KL
                for pulseSize in pulseSizes:
                    pulseDir = f'{tempDir}/{pulseSize}'
                    if not os.path.exists(pulseDir):
                        os.mkdir(pulseDir)
                    for n in model.graph.nodes():
                        pulse        = {n : pulseSize}
                        model.nudges = pulse
                        conditional, px, mi = infcy.runMC(model, snapshots, deltas, repeats)

                        print(f'{datetime.datetime.now().isoformat()} Computing MI')

                        # snapshots, conditional, mi = infcy.reverseCalculation(nSamples, model, deltas, pulse)[-3:]
                        fileName = f"{pulseDir}/{datetime.datetime.now().isoformat()}"
                        fileName += "".join(f"_{key}={settings.get(key, '')}" for key in props)
                        fileName += f'_pulse={pulse}'
                        sr       = SimulationResult(\
                                                mi          = mi,\
                                                conditional = conditional,\
#                                                graph       = model.graph,\
                                                px          = px,\
                                                snapshots   = snapshots)
                        IO.savePickle(fileName, sr)
