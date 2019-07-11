from Toolbox import infcy
from Models.fastIsing import Ising
import networkx as nx
g = nx.path_graph(5)
m = Ising(graph = g)
m.updateType = 'single'
m.t = .2

test = infcy.testSeed(m, N = 4, nSamples = int(1e4))

print(test.shape)
test = test.reshape(test.shape[0], -1)

import matplotlib.pyplot as plt, numpy as np
x = np.arange(m.nNodes + 1)
fig, ax = plt.subplots(4)
for idx, i in enumerate(test):
    ax[idx].hist(i, bins = x, alpha = .2)
    ax[idx].set(yscale = 'log')
print(test[:, :5])
fig.show()
plt.show()
