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

m = Ising(graph, temperature = inf)
m.updateType = 'async'
m.magSide    = 'pos'
print(np.asarray(m.nodeids))

# %%
temps = linspace(0, 1, 3)
x = []

#y = asarray(m.burnin(samples = 1000))

s = time()
for i in range(10):
    m.simulate(10000)
print(time()-s)
# x = array(x)
# plot(x.mean(1))
# %%
