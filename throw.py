import faulthandler
faulthandler.enable()
import networkx as nx
from numpy import *
from matplotlib.pyplot import *
import plotting as plotz

style.use('seaborn-poster')
dataDir = 'Psycho' # relative path careful
import IO
df    = IO.readCSV(f'{dataDir}/Graph_min1_1.csv', header = 0, index_col = 0)
h     = IO.readCSV(f'{dataDir}/External_min1_1.csv', header = 0, index_col = 0)
 #
graph   = nx.from_pandas_adjacency(df)
colors = cm.tab20(arange(graph.number_of_nodes()))
# swap default colors to one with more range
rcParams['axes.prop_cycle'] = cycler('color', colors)


attr = {}
for node, row in h.iterrows():
    attr[node] = dict(H = row['externalField'], nudges = 0)

nx.set_node_attributes(graph, attr)
graph = nx.barabasi_albert_graph(3, 2)
# graph = nx.path_graph(10)
import fastIsing
#graph = nx.path_graph(20)
s = time.process_time()
m = fastIsing.Ising(graph, temperature = 10)
from time import time
m.updateType = 'single'

m.magSide    = 'neg'
m.reset()
import infcy
s = time()
temps = linspace(0, 10, 100)
#mags, sus = m.matchMagnetization(temps, 100, 0)
#fig, ax = subplots()
#ax.scatter(temps, mags)


xx = infcy.getSnapShots(m, 1000, step = 100)
repeats = 10000
deltas = 3
y  = infcy.monteCarlo(m, xx, deltas, repeats)



# for k in y:
    # print(k)
px, mi= infcy.mutualInformation(y, deltas, xx, m )



# %%

d = nansum(log2(px ) * px, -1)
plot(mi, label = 'test', alpha = 0.5 )
#plot(tt - mi, '--', label = 'testest')
#plot(-mi - tt)
#plot(d)
#plot(px[..., 0])
legend()
show()
# fig, ax = subplots()
# for idx, i in enumerate(mi.base.T):
#     print(i)
#     ax.plot(arange(len(i)), i, color = colors[idx], label = m.rmapping[idx]) #, colors[idx], label = m.rmapping[idx])
# ax.legend()
# %%
# from scipy import integrate, optimize
# f = lambda x, a, b, c, d, e: a + b * exp(-c *x) + d * exp(-e * x)
# xx = arange(mi.shape[0])
# fig, ax = subplots(figsize = (10,10))
# d = nx.degree(m.graph)
# for dd, k in dict(d).items():
#     idx = m.mapping[dd]
#     x = mi.base[:, idx]
#     a, b = optimize.curve_fit(f, xx, x)
#     auc, _ = integrate.quad(f, 0, np.inf, args = tuple(a))
#
#     print(auc, dd)
#     ax.scatter(k, auc, color = colors[idx])
#     ax.scatter(k, px.base[-1, idx, :].max(), color = colors[idx], alpha = .2)
#
# idx = px.base[-1, :, :].argmax(axis = 0)
#

# %%

#print(m.graph.degree(m.rmapping[idx[0]]))
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
