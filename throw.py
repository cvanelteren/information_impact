import matplotlib as mpl, matplotlib.pyplot as plt
from Models import  fastIsing, potts
import networkx as nx, numpy as np

n = 30
g = nx.grid_2d_graph(n, n)

#m = fastIsing.Ising(g)

m = potts.Potts(g, temperature = 0, updateType = 'sync', agentStates = [0, 1])
temps = np.linspace(0, 3)

fig, ax = plt.subplots()
for t in temps:
    m.t = t
    m.reset()
    res = m.simulate(1000)
    res = np.array([m.siteEnergy(i) for i in res])
    ax.scatter(t, -res.mean())
fig.show()



