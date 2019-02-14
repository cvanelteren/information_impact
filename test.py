#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 11 08:55:35 2019

@author: casper
"""

import networkx as nx
import msgpack
from numpy import *
from matplotlib.pyplot import *
from Models.fastIsing import Ising

g = nx.barabasi_albert_graph(20, 10)
temps = linspace(0, 100, 100)

m = Ising(g)

mags, sus = m.matchMagnetization(temps)

fig, ax = subplots()
ax.plot(temps, mags, temps, sus)
fig.show()
show()
