
import matplotlib as mpl, matplotlib.pyplot as plt
from Models import  fastIsing, potts
import networkx as nx, numpy as np

n = 500
g = nx.grid_2d_graph(n, n)
g = nx.path_graph(3)
m = fastIsing.Ising(graph = g)



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
while True:
    snapshots    = infcy.getSnapShots(m, 1000)
    conditional, px, mi = infcy.runMC(m, snapshots, 10, 100)

#
##print(m.memory.base)
##print(m.memory.base)
#dtype = int
#fig, ax = plt.subplots()
#tax = ax.twinx()
#ax.plot(temps, a )
#tax.plot(temps, b, 'r')
#fig.show()


#fig, ax = plt.subplots()
#r = []
#for t in temps:
#    m.t = t
#    m.reset()
#    res = m.simulate(n)
#    res = np.array([m.siteEnergy(i) for i in res])
#    r.append(res.sum() / (n * 30 * 30))
##    ax.scatter(t, res)
#ax.plot(temps, r)
##m = fastIsing.Ising(g)
fig, ax = plt.subplots()
tax = ax.twinx()
ax.plot(temps, a)
tax.plot(temps, b, 'r')
fig.show()


plt.show()
# %%
