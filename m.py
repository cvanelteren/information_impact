
# from Utils import IO


import numpy as np, networkx as nx
from Models import FastIsing as FS

modelSettings = dict(\
magSide    = 'neg', \
updateType = '0.25',\
nudgeType  = 'constant',\
)


settings = dict(\
model = FS.Ising,\
repeats = int(1e4),\
deltas = 30,\
step   = int(1e3),\
nSamples = int(1e4),\
trials   = 50,\
burninSamples = 0,\
modelSettings = modelSettings,\
tempres = 100,\
)
for i in range(10):
    r = np.random.uniform(.2, .8)
    g =  nx.erdos_renyi_graph(10, r)
    # m = settings.get('model')(g)
    m = FS.Ising(g)
    m.matchMagnetization()
