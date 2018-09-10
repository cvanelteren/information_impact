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
if __name__ == '__main__':
    # graph = nx.path_graph(12, nx.DiGraph())
    # graph = nx.read_edgelist(f'{os.getcwd()}/Data/bn/bn-cat-mixed-species_brain_1.edges')
#
    dataDir = 'Psycho' # relative path careful
    df    = IO.readCSV('{}/Graph_min1_1.csv'.format(dataDir), header = 0, index_col = 0)
    h     = IO.readCSV('{}/External_min1_1.csv'.format(dataDir), header = 0, index_col = 0)
#
    graph   = nx.from_pandas_adjacency(df) # weights done, but needs to remap to J or adjust in Ising
    #
    attr = {}
    # for node, row in h.iterrows():
        # attr[node] = dict(H = row['externalField'], nudges = 0)
    # nx.set_node_attributes(graph, attr)
    # for i, j in graph.edges():
        # graph[i][j]['weight'] = 1 # sign(graph[i][j]['weight'])
    ## prisoner graph
    # tmp = 'weighted_person-person_projection_anonymous_combined.graphml'
    # fn  = f'{os.getcwd()}/Data/bn/{tmp}'
    # graph = nx.read_graphml(fn)
    # from time import time
    # s = time()
    # print(time() - s)
    # g = graph.copy()
    # theta = 40
    # for i in graph.nodes():
    #     if graph.degree(i) < theta:
    #         g.remove_node(i)
    # graph = g
    # nx.set_edge_attributes(graph, 1, 'weight') # set this to off
    # assert 0
    graph = nx.florentine_families_graph()
    # for i, j in graph.edges():
    #     graph[i][j]['weight'] = 1
    # # %%


    now = datetime.datetime.now()
    targetDirectory = f'{os.getcwd()}/Data/{now}'
    os.mkdir(targetDirectory)
    # %%
    # graph = nx.karate_club_graph()
    # graph = nx.grid_2d_graph(10,10, periodic = True)

    # s = nx.utils.powerlaw_sequence(100, 1.6) #100 nodes, power-law exponent 2.5\n",
    # graph = nx.expected_degree_graph(s, selfloops=False)
    # graph = nx.krackhardt_kite_graph()
    # graph = nx.frucht_graph()
    # graph = nx.karate_club_graph()
    # graph = nx.path_graph(5, nx.DiGraph())
    # graph = nx.path_graph(5)
    # graph = nx.florentine_families_graph()
    # print(graph.number_of_nodes())
    # graph = nx.barabasi_albert_graph(10, 4)
    for i in range(1):
        # graph = nx.barabasi_albert_graph(10, 3)
        model = fastIsing.Ising(graph = graph, \
                                temperature = 0, \
                                updateMethod = 'glauber', mode = 'sync')
        kSamples = 200
        deltas   = 5
        step     = 10
        nSamples = 100000
        burninSamples = 5
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
            magRange = linspace(.9, .8, 5) # .5 to .2 seems to be a good range; especially .2
            magRange = array([.9])
            temps = linspace(0, 10, 100)

            temps, mag, sus = model.matchMagnetization(  temps = temps,\
             n = 1000, burninSamples = 0)

            # func = lambda x, a, b, c, d :  a / (1 + exp(b * (x - c))) + d # tanh(-a * x)* b + c
            func = lambda x, a, b, c : a + b*exp(-c * x)
            a, b = scipy.optimize.curve_fit(func, temps, mag, maxfev = 10000)
            # g = gp(normalize_y = True)
            # g.fit(temps[:, None], mag)
            # fff    = lambda x, b : g.predict(x[:, None]) - b

            # run the simulation per temperature
            temperatures = array([])
            f_root = lambda x,  c: func(x, *a) - c
            magRange *= max(mag)
            for m in magRange:
                r = scipy.optimize.root(f_root, 0, args = (m), method = 'linearmixing')

                temperatures = hstack((temperatures, abs(r.x)))

            fig, ax = subplots()
            # ax.scatter(temps, func(temps, *a))
            xx = linspace(0, max(temps), 1000)
            ax.plot(xx, func(xx, *a))
            # ax.plot(temps, g.predict(temps[:, None]))
            ax.scatter(temperatures, magRange, c ='red')
            ax.scatter(temps, mag, alpha = .2)
            setp(ax, **dict(xlabel = 'Temperature', ylabel = '<M>'))
            savefig(f'{targetDirectory}/temp vs mag.png')
            # show()
            tmp = dict(temps = temps, \
            temperatures = temperatures, magRange = magRange, mag = mag)
            IO.savePickle(f'{targetDirectory}/mags.pickle', tmp)
        # assert 0
        # temperatures = [800]

        # state = zeros(12)
        # n = 12
        # N = 2**n
        # state = zeros(n)
        # snapshots = {}
        # for i in tqdm(range(N)):
        #     string = format(i, '012b')
        #     for idx, j in enumerate(string):
        #         state[idx] = 1 if j == '1' else -1
        #     snapshots[tuple(state)] = 1 / N
        for t in temperatures:
            print(f'Setting {t}')
            model.t = t
            # if os.path.isfile(fileName):
                # if input('Do you want to remove this file Y/N') == 'Y':
                    # os.remove(fileName)
            # s = time()
            # if os.path.isfile(fileName) == False:
        #        mis = {}
            from multiprocessing import cpu_count
            # st = [random.choice(model.agentStates, size = model.nNodes) for i in range(nSamples)]
            print('Getting snapshots')
            s = infcy.getSnapShots(model, nSamples, \
                                           parallel     = cpu_count(), \
                                           burninSamples = burninSamples, \
                                           step = step)
            snapshots = s
            # match magnetization
            # snapshots = {}
            # for i, j in tqdm(s.items()):
            #     i = array(i)
            #     if sign(mean(i)) < 0:
            #         i *= -1
            #     snapshots[tuple(i)] = snapshots.get(tuple(i), 0) + j


            pulseSize = 1
            pulses = {node : pulseSize for node in model.graph.nodes()}
            pulse  = {}
            joint = infcy.monteCarlo_alt(model = model, snapshots = snapshots,\
                                            deltas = deltas, kSamples = kSamples,\
                                            mode = mode, pulse = pulse)
            # joint = infcy.monteCarlo(model = model, snapshots = snapshots, conditions = conditions,\
            # deltas = deltas, kSamples = kSamples, pulse = pulse, mode = 'source')
            print('Computing MI')
            mi = infcy.mutualInformation_alt(joint)
            # mi   = infcy.mutualInformation(joint, conditions, deltas)
            fileName = f'{targetDirectory}/{time()}_nSamples={nSamples}_k={kSamples}_deltas={deltas}_mode_{mode}_t={t}_n={model.nNodes}_pulse={pulse}_{model.updateMethod}.pickle'
            IO.savePickle(fileName, dict(mi = mi, joint = joint, model = model,\
                                            snapshots = snapshots))

            for n, p in pulses.items():
                pulse = {n : p}
                joint = infcy.monteCarlo_alt(model = model, snapshots = snapshots,\
                                        deltas = deltas, kSamples = kSamples,\
                                        mode = mode, pulse = pulse)
                print('Computing MI')
                print(type(joint))
                mi = infcy.mutualInformation_alt(joint)
                fileName = f'{targetDirectory}/{time()}_nSamples={nSamples}_k={kSamples}_deltas={deltas}_mode_{mode}_t={t}_n={model.nNodes}_pulse={pulse}_{model.updateMethod}.pickle'
                IO.savePickle(fileName, dict(mi = mi, joint = joint, model = model,\
                                        snapshots = snapshots))

                # print(f'{(time() - s)/60} elapsed minutes')
