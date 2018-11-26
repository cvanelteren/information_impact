#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 26 20:54:35 2018

@author: casper
"""

from fastIsing import Ising
import networkx as nx
from numpy import *


graph = nx.path_graph(3)
model = Ising(graph = graph, temperature = 1)
for i in a:
    a = model.updateState(model.sampleNodes[model.mode](model.nodeIDs))
    print(a)