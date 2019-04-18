import matplotlib as mpl
from Models import  fastIsing, potts
import networkx as nx, numpy as np

n = 30
g = nx.grid_2d_graph(n, n)

#m = fastIsing.Ising(g)
m = potts.Potts(g, temperature = .5)

res = m.simulate(190)


import matplotlib.pyplot as plt
print(res)
fig, ax = plt.subplots()
ax.imshow(res[-1].reshape(n, n), aspect = 'auto')
fig.show()
