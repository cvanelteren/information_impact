import matplotlib.pyplot as plt, cmasher as cmr, pandas as pd
import numpy as np, os, sys, networkx as nx, warnings
from plexsim import models
from imi import infcy

warnings.simplefilter("ignore")
plt.style.use("fivethirtyeight spooky".split())

g = nx.grid_graph((6, 6))
g = nx.krackhardt_kite_graph()
g = nx.karate_club_graph()
g = nx.path_graph(1000)

print(g)
m = models.Potts(g, t=5)

# print(m.sampleNodes(1).base)
N = 10000
M = 10000
import time


# print(m.simulate(M).shape)
sim = infcy.Simulator(m)

s = time.time()
snaps = sim.snapshots(N)
print(time.time() - s)
print("retrieved snaps")

s = time.time()
S = sim.forward(snaps, repeats=M, time=np.arange(10), n_jobs=1)["conditional"]
print(time.time() - s)
px, mi = infcy.mutualInformation(S, snaps)
fig, ax = plt.subplots()
ax.plot(mi)
fig.show()
plt.show(block=1)

print(mi.sum(0))
