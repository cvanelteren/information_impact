#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  6 13:57:13 2018

@author: casper
"""

import fastIsing
import networkx as nx,  numpy as np, matplotlib.pyplot as plt

graph = nx.path_graph(4)

m = fastIsing.Ising(graph, 1000)

import copy
x = [m for i in range(10)]
def func(x):
#    print(id(x))
#    print(mp.current_process())
    for i in range(10000):
        x.updateState(x.nodeids)
    return x.states
import multiprocessing as mp

xx = np.array([m.states for m in x])
print(np.unique(xx, axis = 0).shape)
with mp.Pool(4)  as p:
   y =  np.asarray(p.map(func, x))
   
print(np.unique(y, axis = 0).shape)
