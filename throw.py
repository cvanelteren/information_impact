
import matplotlib as mpl, matplotlib.pyplot as plt
from Models import  fastIsing, potts
import networkx as nx, numpy as np

import time
n = 500
# g = nx.grid_2d_graph(n, n)
# g = nx.star_graph(10)

# g = nx.path_graph(3)
#g.add_edge(0,0)
g =  nx.soft_random_geometric_graph(20, .5)
#g = nx.grid_2d_graph(10, 10)
#g = nx.path_graph(3, nx.DiGraph()).
#g = nx.barabasi_albert_graph(10, 3)
#g.add_edge(0, 0)

#g = nx.barabasi_albert_graph(10, 2)

m = fastIsing.Ising(graph = g, updateType = 'async', \
                    magSide = 'neg', \
                    nudgeType = 'constant')
temps = np.linspace(0, g.number_of_nodes(), 100)
mag  = m.matchMagnetization(temps, 100)[0]
idx = np.argmin(abs(mag - .7))

fig, ax = plt.subplots()
ax.plot(temps, mag)
ax.plot(temps[idx], mag[idx], 'r.')

m.t = temps[idx]
#m.t = 1
#print(m.states.base)


colors = plt.cm.tab20(np.arange(m.nNodes))

# m = potts.Potts(graph = g, temperature = 0, \
#                 updateType = 'async', \
#                 agentStates = [1, 2, 3], \
#                 memorySize = 0, delta = 0.5)

## print(m.memorySize, m.memory.base)
temps = np.linspace(0, 10, 300)
#temps = np.logspace(-10, 1, 300)
##
N = 100
##print(m.memory.base.shape)
# a, b = m.matchMagnetization(temps, N)

from Toolbox import infcy


print('>', m.nudges.base)
#assert 0
#
#m.nudges = {'0': 1}

deltas = 20
start = time.time()
snapshots    = infcy.getSnapShots(m, int(1e2), 1)

repeats = int(1e3)
conditional, px, mi = infcy.runMC(m, snapshots, deltas, repeats)

assert len(conditional) == len(snapshots)
from Utils.stats import KL, hellingerDistance

# %%

NUDGE = 1
out = np.zeros((m.nNodes, deltas))
for node, idx in m.mapping.items():
    m.nudges = {node : NUDGE}
    c, p, n = infcy.runMC(m, snapshots, deltas, repeats)
    out[idx, :] = KL(px, p).sum(-1)
print(time.time() - start)
# %%
fig, ax = plt.subplots()
[ax.plot(i, color = colors[idx], label = m.rmapping[idx]) for idx, i in enumerate(out)]
#ax.set_xlim(deltas // 2 - 2, deltas)
#ax.set_yscale('log')
ax.legend()
fig.show()
# %%
fig, ax = plt.subplots()
nx.draw(g, ax = ax, with_labels = 1)
fig.show()
plt.show()
