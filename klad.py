import numpy as np, tempfile, h5py
import information as i, information2 as ii, networkx as nx, os
from fastIsing import Ising
import time

# %%
import os
t = 30
graph = nx.read_edgelist(f'{os.getcwd()}/Data/bn/bn-cat-mixed-species_brain_1.edges')
# graph = nx.path_graph(2, nx.DiGraph())
model = Ising(graph, t, doBurnin = False)

nSamples = int(10000)
repeats = 10000
deltas   = 10
d = dict(model = model, repeats = repeats, deltas = deltas, pulse = {})
fileName = f'/media/casper/375C5C665C66D4FA/tester.h5'
fileName = f'{os.getcwd()}/cat_test_{t}.h5'
#
#ii.getSnapshots(fileName = fileName, model = model, nSamples = nSamples)
#ii.montecarlo(fileName = fileName, **d)
#ii.mutualInformationShift(model, fileName, repeats, mode = 'source')
# %%
from h5py import File
import  os
from numpy import *
from matplotlib.pyplot import *
with File(fileName) as f:
    for i in f:
        print(i, f[i].shape)
#    mi = f['mi'].value
    mc = f['mc'].value
    snapshots = f['snapshots'].value
    joint  = f['joint'].value
#print(mi)
#fig, ax = subplots(); ax.plot(mi)
#show()
