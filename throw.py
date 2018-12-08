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
#graph = nx.gnp_random_graph(10, 0.2)
graph  = nx.path_graph(3)
# graph = nx.path_graph(10)
import fastIsing
#graph = nx.path_graph(20)
s = time.process_time()
fig, ax = subplots()
nx.draw(graph, with_labels = 1)
m = fastIsing.Ising(graph, temperature = 1)
from time import time
m.updateType = 'single'

m.magSide    = 'pos'
#m.reset()
import infcy
s = time()
temps = linspace(0, 10, 100)
mags, sus = m.matchMagnetization(temps, 10000, 0)
fig, ax = subplots()
ax.scatter(temps, mags)
#
xx = infcy.getSnapShots(m, 1000, step = 100)
repeats = 100000
deltas = 3
y  = infcy.monteCarlo(m, xx, deltas, repeats)
px, mi= infcy.mutualInformation(y, deltas, xx, m )
fig, ax = subplots()
ax.plot(mi)
