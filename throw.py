
import matplotlib as mpl, matplotlib.pyplot as plt
from Models import  fastIsing, potts
import networkx as nx, numpy as np

n = 10
g = nx.grid_2d_graph(n, n)

m = fastIsing.Ising(graph = g)


#assert 0

m = potts.Potts(graph = g, temperature = 0, \
                updateType = 'async', \
                agentStates = [0, 1, 2], \
                memorySize = 3)

## print(m.memorySize, m.memory.base)
temps = np.linspace(0, 10, 30)
#temps = np.logspace(-10, 1, 30)
##
N = 1000
#
##print(m.memory.base.shape)
a, b = m.matchMagnetization(temps, N)
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

