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
import IO, multiprocessing as mp
import datetime
from artNet import Net
from time import time
close('all')
np.random.seed() # set seed
if __name__ == '__main__':
    # graph = nx.path_graph(12, nx.DiGraph())
    # graph = nx.read_edgelist(f'{os.getcwd()}/Data/bn/bn-cat-mixed-species_brain_1.edges')
    repeats       = 1000
    deltas        = 20
    step          = 1
    nSamples      = 1000
    burninSamples = 5
    pulseSize     = 1

    numIter       = 1
    magSide       = 'neg'
    CHECK         = .8

    dataDir = 'Psycho' # relative path careful
    df    = IO.readCSV('{}/Graph_min1_1.csv'.format(dataDir), header = 0, index_col = 0)
    h     = IO.readCSV('{}/External_min1_1.csv'.format(dataDir), header = 0, index_col = 0)
#
    graph   = nx.from_pandas_adjacency(df) # weights done, but needs to remap to J or adjust in Ising
    #
    attr = {}
    for node, row in h.iterrows():
        attr[node] = dict(H = row['externalField'], nudges = 0)
    nx.set_node_attributes(graph, attr)

    graph = nx.krackhardt_kite_graph()
    # graph = nx.path_graph(3, nx.DiGraph())
    # graph = nx.path_graph(3)
    # graph = nx.star_graph(4)
    # graph = nx.watts_strogatz_graph(20, 2, .3)
    now = datetime.datetime.now()
    targetDirectory = f'{os.getcwd()}/Data/{now}'
    os.mkdir(targetDirectory)

    for i in range(numIter):
        # graph = nx.barabasi_albert_graph(10, 3)
        model = fastIsing.Ising(graph = graph, \
                                temperature = 0, \
                                mode = 'async', magSide = magSide)


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
            magRange = linspace(.95, .8, 2) # .5 to .2 seems to be a good range; especially .2
            magRange = array([CHECK])
            # magRange = array([.9, .2])
            temps = linspace(.1, 10, 20)

            temps, mag, sus = model.matchMagnetization(  temps = temps,\
             n = 100, burninSamples = 0)

            func = lambda x, a, b, c, d :  a / (1 + exp(b * (x - c))) + d # tanh(-a * x)* b + c
            # func = lambda x, a, b, c : a + b*exp(-c * x)
            a, b = scipy.optimize.curve_fit(func, temps, mag, maxfev = 10000)
            # g = gp(normalize_y = True)
            # g.fit(temps[:, None], mag)
            # fff    = lambda x, b : g.predict(x[:, None]) - b

            # run the simulation per temperature
            temperatures = array([])
            f_root = lambda x,  c: func(x, *a) - c
            magRange *= max(mag)
            for m in magRange:
                r = scipy.optimize.root(f_root, 0, args = (m), method = 'linearmixing')#, method = 'linearmixing')
                rot = r.x if r.x > 0 else 0
                temperatures = hstack((temperatures, rot))

            fig, ax = subplots()
            # ax.scatter(temps, func(temps, *a))
            xx = linspace(0, max(temps), 1000)
            ax.plot(xx, func(xx, *a))
            # ax.plot(temps, g.predict(temps[:, None]))
            ax.scatter(temperatures, func(temperatures, *a), c ='red')
            ax.scatter(temps, mag, alpha = .2)
            setp(ax, **dict(xlabel = 'Temperature', ylabel = '<M>'))
            savefig(f'{targetDirectory}/temp vs mag.png')
            # show()
            tmp = dict(temps = temps, \
            temperatures = temperatures, magRange = magRange, mag = mag)
            IO.savePickle(f'{targetDirectory}/mags.pickle', tmp)
        # assert 0
        # temperatures = [800]

        for t in temperatures:
            print(f'Setting {t}')
            model.t = t

            from multiprocessing import cpu_count
            # st = [random.choice(model.agentStates, size = model.nNodes) for i in range(nSamples)]
            print('Getting snapshots')

            snapshots = infcy.getSnapShots(model, nSamples, \
                                           parallel     = cpu_count(), \
                                           burninSamples = burninSamples, \
                                           step = step)



            pulses = {node : pulseSize for node in model.graph.nodes()}
            pulse  = {}
            conditional = infcy.monteCarlo_alt(model = model, snapshots = snapshots,\
                                           deltas = deltas, repeats = repeats,\
                                           mode = mode, pulse = pulse)

            # px, conditional, snapshots, mi = infcy.reverseCalculation(nSamples, model, deltas, pulse)[-4:]
            # conditional = infcy.monteCarlo(model = model, snapshots = snapshots, conditions = conditions,\
             # deltas = deltas, repeats = repeats, pulse = pulse, mode = 'source')

            print('Computing MI')
            cpx, mi = infcy.mutualInformation_alt(conditional, deltas, snapshots, model)
            # mi   = array([infcy.mutualInformation(joint, condition, deltas) for condition in conditions.values()])

            fileName = f'{targetDirectory}/{time()}_nSamples={nSamples}_k={repeats}_deltas={deltas}_mode_{mode}_t={t}_n={model.nNodes}_pulse={pulse}.pickle'
            IO.savePickle(fileName, dict(
            mi = mi, conditional = conditional, model = model,\
            px = cpx, snapshots = snapshots))

            for n, p in pulses.items():
                pulse = {n : p}
                conditional = infcy.monteCarlo_alt(model = model, snapshots = snapshots,\
                                        deltas = deltas, repeats = repeats,\
                                        mode = mode, pulse = pulse)
                print('Computing MI')
                px, mi = infcy.mutualInformation_alt(conditional, deltas, snapshots, model)
                # snapshots, conditional, mi = infcy.reverseCalculation(nSamples, model, deltas, pulse)[-3:]
                fileName = f'{targetDirectory}/{time()}_nSamples={nSamples}_k={repeats}_deltas={deltas}_mode_{mode}_t={t}_n={model.nNodes}_pulse={pulse}.pickle'
                IO.savePickle(fileName, dict(
                mi = mi, conditional = conditional, model = model,\
                px = px, snapshots = snapshots))
                print(n)
                print(px[1, model.mapping[n], :], '\n', cpx[1, model.mapping[n], :])
                # print(f'{(time() - s)/60} elapsed minutes')
