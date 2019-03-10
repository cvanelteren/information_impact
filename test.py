#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 11 08:55:35 2019

@author: casper
"""

import networkx as nx
#import msgpack
#from numpy import *
import matplotlib.pyplot as plt
import numpy as np

def plotDouble(data, ax):
    assert data.shape[0] == 2
    ax.scatter(*data, marker = 4, s = s)
    ax.scatter(*data, marker = 5, s = s)
    ax.scatter(*data, marker = 6, s = s)
    ax.scatter(*data, marker = 7, s = s)
    
fig, ax  = plt.subplots()
plotDouble(np.random.rand(2, 10), ax)
fig.show()