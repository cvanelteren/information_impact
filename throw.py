
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
graph = nx.barabasi_albert_graph(10, 5)



# print(m.sampleNodes(10))
#print(m.graph)
#p = m.mapping
#m.simulate(pulse = p)
from time import time
from fastIsing import Ising

m = Ising(graph, temperature = 2)
m.updateType = 'async'
m.magSide    = 'neg'

# %%
temps = linspace(0, 1, 3)
x = []

#y = asarray(m.burnin(samples = 1000))



temps = linspace(0, 10, 59)
a = []
# for t in temps:
#     m.states = 1
#     m.t = t
#     x = np.asarray(m.simulate(10000))
#     a.append(abs(x.mean()))
# # print(time()-s)
# fig, ax = subplots()
# ax.plot(temps,a)
# ax.set_xscale('log')
print(a)

import infcy
s = time()
from pathos import multiprocessing as mp
from copy import copy
from functools import partial
x = [copy(m) for i in range(100)]
func = partial(infcy.getSnapShots, nSamples = 100)
with mp.Pool(4) as p:
    p.map(func, x)
# x  = infcy.getSnapShots(m, 10000)
# y  = infcy.monteCarlo_alt(m, x, repeats = 10000)

for k, v in y.items():
    print(v.max())
j = 0
print(time() - s)
show()
# x = array(x)
# plot(x.mean(1))
# %%
