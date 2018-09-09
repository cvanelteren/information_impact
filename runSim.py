#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 20 16:16:14 2018

@author: casper
"""

from numpy import *
from matplotlib.pyplot import *
import IO, information, fastIsing, networkx as nx, plotting as plotz, pickle, os
# %%
if __name__ == '__main__':

    # %%
    dataDir = 'Psycho' # relative path careful
    df    = IO.readCSV('{}/Graph_min1_1.csv'.format(dataDir), header = 0, index_col = 0)
    h     = IO.readCSV('{}/External_min1_1.csv'.format(dataDir), header = 0, index_col = 0)

    graph   = nx.from_pandas_adjacency(df) # weights done, but needs to remap to J or adjust in Ising
    # %%
    attr = {}
    for node, row in h.iterrows():
        attr[node] = dict(H = row['externalField'], nudges = 0, state = 1 if random.rand() < .5 else -1)
    nx.set_node_attributes(graph, attr)
    # nx.set_edge_attributes(graph, nx.get_edge_attributes(graph, 'weight'), 'J')
    try:
        with open('psycho_temp', 'rb') as f:
            t = pickle.load(f) * 2
    except:
        temperatures = logspace(-1, 0,  10)
        temperatures = linspace(.2, .3, 5)
        a = fastIsing.matchTemperature(graph = graph, temperatures = temperatures, nSamples = 10000_000, step = 1)
        t = a[2]
        with open('psycho_temp', 'w') as f:
            pickle.dump(t, f)
    # graph = nx.read_edgelist(f'{os.getcwd()}/Data/bn/bn-cat-mixed-species_brain_1.edges')
    model = fastIsing.Ising(graph, t, doBurnin = False)

    effect = inf
    stdRuns = {'nSamples' : 1_00_000  , 'step' : 1, 'deltas': 10, 'kSamples': 10000, 'pulse' : True}
    # stdRuns = {'nSamples' : 1000, 'step' : 1, 'deltas': 1, 'kSamples': 1, 'pulse' : True}
    paramString = ''.join(f'{key}={value}_' for key, value in stdRuns.items())
    storageName = paramString +  f'nudge_size={effect}_' + f'T={t}' + f'_N={model.nNodes}_'
    # get results no nudge and every nudge
    # %%
    def runNudge(model, storageName, dirName):
        tmpList = ['control', *model.graph.nodes()]
        for idx, node in enumerate(tmpList):
            name = storageName + str(node) + '.pickle'
            if name not in os.listdir(dirName):
                print(storageName + str(node))
                if node == 'control':
                    nudgeInfo = {}
                else:
                    nudgeInfo = {node : array([effect])}
                l = information.nudgeOnNode(nudgeInfo,  model,\
                       mode = 'source', **stdRuns)
                IO.savePickle(dirname + storageName + str(node), l)
                del l
            else:
                print(f'skipping {node}')
    def runTemp(model, temperatures, parameters, dirName = os.getcwd()):
        name = ''.join(f'{key}={value}_' for key, value in parameters.items())
        from tqdm import tqdm
        for t in tqdm(temperatures):
            store = name + f'T={t:1.4e}_N={model.nNodes}.pickle'
            if store not in os.listdir(dirName):
                # print(store); assert 0
                model.beta = inf if t == 0 else 1/t
                l = information.nudgeOnNode({}, model, mode = 'source', **parameters,
                                            parallel = 4)
                IO.savePickle(dirName + store, l)
                del l
            else:
                print(f'skipping{t}')

    temperatures = linspace(.1, .11, 2)
    temperatures = [10]
    # dd = '/media/casper/375C5C665C66D4FA/'
    dd = os.getcwd()
    runTemp(model, temperatures, stdRuns, dirName =  dd)
    #%%
    fitter = lambda x, a, b, c: a * exp(-b * (x - c))
    func = lambda x, a, b, c, d, e, f, g, h, i:  fitter(x, a, b, c)  \


    colors = cm.tab20(arange(model.nNodes))
    for file in os.listdir(dd):
        if file.endswith('65.pickle'):
            tmp = float(file.split('_')[-2].split('=')[-1])
            print(file)
            I = IO.loadPickle(dd + file)['I']
            fig, ax = subplots();
            # ax.plot(res['I'], '.')
            print(I)
            [ax.plot(i, '.', color = j) for i, j in zip(I.T, colors)]
#                plotz.showFit(model, res['I'].T, func)
#                ax = gca()
            ax.set_title(file)
    show()
