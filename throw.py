import faulthandler
faulthandler.enable()
import networkx as nx
from numpy import *
from matplotlib.pyplot import *
style.use('seaborn-poster')
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
m.updateType = 'async'
m.magSide    = 'neg'

import infcy
s = time()
temps = linspace(0, 10, 1000)
mags  = empty(temps.size)
#sys.settrace(trace)
xx = infcy.getSnapShots(m, 100)
deltas = 5
#while True:
y  = infcy.monteCarlo(m, xx, repeats = 1000, deltas = deltas)
#     for i, t in enumerate(temps):
#         m.t = t
#         m.reset()
#         # print('>', m.states.base)
#         mags[i] = abs(np.mean(m.simulate(100)))
    # print(m.states.base)

m.t = .5


y  = infcy.monteCarlo(m, xx, repeats = 1000, deltas = deltas)
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

# %%


from scipy import integrate, optimize
f = lambda x, a, b, c : a + b * exp(-c *x)
xx = arange(mi.shape[0])
fig, ax = subplots(figsize = (10,10))
d = nx.degree(m.graph)
for dd, k in dict(d).items():
    idx = m.rmapping[dd]
    x = mi.base[:, idx]
    a, b = optimize.curve_fit(f, xx, x)
    auc, _ = integrate.quad(f, 0, np.inf, args = tuple(a))

    print(auc, dd)
    ax.scatter(auc, dd)
    ax.scatter(px.base[-1, idx, :].max(), dd, alpha = .2)

idx = px.base[-1, :, :].argmax(axis = 0)

print(m.graph.degree(m.rmapping[idx[0]]))
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
