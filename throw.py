
import networkx as nx
from numpy import *
from matplotlib.pyplot import *

dataDir = 'Psycho' # relative path careful
import IO
df    = IO.readCSV(f'{dataDir}/Graph_min1_1.csv', header = 0, index_col = 0)
h     = IO.readCSV(f'{dataDir}/External_min1_1.csv', header = 0, index_col = 0)
 #
graph   = nx.from_pandas_adjacency(df)

attr = {}
for node, row in h.iterrows():
    attr[node] = dict(H = row['externalField'], nudges = 0)
nx.set_node_attributes(graph, attr)
graph = nx.barabasi_albert_graph(10, 2)
# graph = nx.path_graph(10)
import fastIsing

s = time.process_time()
m = fastIsing.Ising(graph, temperature = .5)
from time import time
m.updateType = 'single'
m.magSide    = 'neg'

import infcy
s = time()
from pathos import multiprocessing as mp
from copy import copy
from functools import partial
# x = [copy(m) for i in range(100)]
# func = partial(infcy.getSnapShots, nSamples = 100)
temps = linspace(0, 10, 1000)
mags  = empty(temps.size)

for i, t in enumerate(temps):
    m.t = t
    m.reset()
    # print('>', m.states.base)
    mags[i] = abs(np.mean(m.simulate(100)))
    # print(m.states.base)

m.t = np.inf

xx = infcy.getSnapShots(m, 1000)

#print(m._states)
deltas = 20
y  = infcy.monteCarlo(m, xx, repeats = 10000, deltas = deltas)
print('>')
# for k in y:
    # print(k)
px, mi = infcy.mutualInformation(y, deltas, xx, m )
plot(mi.base)
def decodeState(state, nStates, nNodes):
    tmp = format(state, f'0{(nStates -1 ) * nNodes}b')
    n   = len(tmp)
    nn  = nStates - 1
    return np.array([int(tmp[i : i + nn], 2) for i in range(0, n, nn )], dtype = int) * 2 - 1
#for k, v in xx.items():
#    print(k)
#    print(decodeState(k, 2, 10))

# for k, v in y.items():
#     print(k.base, v.base)
#
# px, mi = infcy.mutualInformation(y, deltas, xx, m, )
# fig, ax = subplots()
# ax.plot(mi)
# ax.plot(temps, mags)
# print(time() - s)
# fig, ax = subplots()
# ax.plot(mi)
# show()
# x = array(x)
# plot(x.mean(1))
# %%


show()
