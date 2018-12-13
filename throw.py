import faulthandler
faulthandler.enable()
import networkx as nx
from numpy import *
from matplotlib.pyplot import *
import plotting as plotz, copy

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
#graph  = nx.path_graph(3)
graph = nx.barabasi_albert_graph(10, 2)

# graph = nx.path_graph(10)
import fastIsing
#graph = nx.path_graph(20)
s = time.process_time()
fig, ax = subplots()
nx.draw(graph, with_labels = 1)
m = fastIsing.Ising(graph, temperature = 1)
import copy

c = copy.deepcopy(m)
c.states = 2

print(id(c.states) == id(m.states))
#assert id(c.states) != id(m.states)
from time import time
m.updateType = 'single'

m.magSide    = 'pos'
###m.reset()
import infcy

temps = linspace(0, 10, 100)
mags, sus = m.matchMagnetization(temps, 1000, 0)
print(m.states)
#fig, ax = subplots()
#ax.scatter(temps, mags)
#fig.canvas.flush_events()
#
xx = infcy.getSnapShots(m, 10000, step = 100)
repeats = 10000
deltas = 20
#while True:
s = time()
y  = infcy.monteCarlo(m, xx, deltas, repeats)
print('Elapsed ', time() - s)
print(m.states)
px, mi= infcy.mutualInformation(y, deltas, xx, m )
fig, ax = subplots()
ax.plot(mi)
show()
# %%
