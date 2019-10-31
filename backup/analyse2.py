#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 16 14:55:01 2019

@author: casper
"""

from Utils import IO, plotting as plot
import numpy as np, matplotlib.pyplot as plt

root = 'Data/1548025318.5751357'



for i in range(10):
    w = nx.utils.powerlaw_sequence(N, 1.6)
    g = nx.expected_degree_graph(w)
    
    g = sorted(nx.connected_component_subgraphs(g), key = lambda x: len(x))[-1]
    fig, ax = plt.subplots()
    nx.draw(g, ax = ax, pos = nx.circular_layout(g))
    deg = list(dict(g.degree()).values())
#    ax.hist(deg, bins = 10)