
import matplotlib as mpl, matplotlib.pyplot as plt
from Models import  fastIsing, potts
import networkx as nx, numpy as np

n = 500
# g = nx.grid_2d_graph(n, n)
# g = nx.star_graph(10)

g = nx.path_graph(3, nx.DiGraph())

m = fastIsing.Ising(graph = g)
print(m.states.base)



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


print(m.nudges.base)


snapshots    = infcy.getSnapShots(m, 1000)
conditional, px, mi = infcy.runMC(m, snapshots, 13, 100)

m.nudges = {0 : 1}
c, p, n = infcy.runMC(m, snapshots, 13, 1000)
#
# %%
from Utils.stats import KL

x  = KL(px, p)

plt.plot(x.sum(-1))
