
import matplotlib as mpl, matplotlib.pyplot as plt
from Models import  fastIsing, potts
import networkx as nx, numpy as np

n = 500
# g = nx.grid_2d_graph(n, n)
# g = nx.star_graph(10)

g = nx.path_graph(5)

#g.add_edge(0, 0)

#g = nx.barabasi_albert_graph(10, 2)

m = fastIsing.Ising(graph = g, updateType = 'single')
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

deltas = 100
snapshots    = infcy.getSnapShots(m, 10000)
repeats = int(1e5)
conditional, px, mi = infcy.runMC(m, snapshots, deltas, repeats)
from Utils.stats import KL

NUDGE = np.inf

out = np.zeros((m.nNodes, deltas))
for node, idx in m.mapping.items():
    m.nudges = {node : NUDGE}
    
    c, p, n = infcy.runMC(m, snapshots, deltas, repeats)
    out[idx, :] = KL(px, p).sum(-1)
# %%
fig, ax = plt.subplots()
[ax.plot(i, color = colors[idx], label = m.rmapping[idx]) for idx, i in enumerate(out)]
ax.set_xlim(deltas // 2, deltas)
ax.legend()
fig.show()
# %%
fig, ax = plt.subplots()
nx.draw(g, ax = ax, with_labels = 1)
fig.show()
    
