#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 11 08:55:35 2019

@author: casper
"""

import networkx as nx
import msgpack
from numpy import *
from Models.fastIsing import Ising

g = nx.path_graph(5)
temps = linspace(0, 10)

m = Ising(g)

m.matchMagnetization(temps)