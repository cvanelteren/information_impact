#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  6 13:57:13 2018

@author: casper
"""

import fastIsing
import networkx as nx,  numpy as np, matplotlib.pyplot as plt

graph = nx.path_graph(4)

m = fastIsing.Ising(graph, 1)

x = [m, m]
def func(x):
    return 0
import multiprocessing as mp

with mp.Pool(3) as p:
    p.map(func, x)