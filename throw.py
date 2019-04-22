import matplotlib as mpl, matplotlib.pyplot as plt
from Models import  fastIsing, potts
import networkx as nx, numpy as np

n = 30
g = nx.grid_2d_graph(n, n)

#m = fastIsing.Ising(g)

m = potts.Potts(g, temperature = 0, updateType = 'async', agentStates = [0, 1])
temps = np.linspace(0, 10, 30)

n = 50
a, b = m.matchMagnetization(temps, n)


fig, ax = plt.subplots()
tax = ax.twinx()
ax.plot(temps, a)
tax.plot(temps, b, 'r')
fig.show()


fig, ax = plt.subplots()
r = []
for t in temps:
    m.t = t
    m.reset()
    res = m.simulate(n)
    res = np.array([m.siteEnergy(i) for i in res])
    r.append(res.sum() / (n * 30 * 30))
#    ax.scatter(t, res)
ax.plot(temps, r)
#m = fastIsing.Ising(g)
#a, b = m.matchMagnetization(temps, n)
#fig, ax = plt.subplots()
#tax = ax.twinx()
#ax.plot(temps, a)
#tax.plot(temps, b, 'r')
#fig.show()


plt.show()
# %%
