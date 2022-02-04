import matplotlib.pyplot as plt, cmasher as cmr, pandas as pd
import numpy as np, os, sys, networkx as nx, warnings
from plexsim import models
from imi import infcy

warnings.simplefilter("ignore")
plt.style.use("fivethirtyeight spooky".split())

g = nx.grid_graph((6, 6))
g = nx.krackhardt_kite_graph()
# g = nx.karate_club_graph()
g = nx.path_graph(1000)

print(g)
m = models.Potts(g, t=0.5, sampleSize=1)
print(m.simulate(1000))

# print(m.sampleNodes(1).base)
N = 100
M = 100

# print(m.simulate(M).shape)
sim = infcy.Simulator(m)
snaps = sim.snapshots(N)
print("retrieved snaps")
s = sim.forward(snaps, repeats=M, time=np.arange(20), n_jobs=1)["conditional"]
px, mi = infcy.mutualInformation(s, snaps)
fig, ax = plt.subplots()
ax.plot(mi)
fig.show()
plt.show(block=1)

print(mi)
