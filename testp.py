from fastIsing import Ising
import matplotlib.pyplot as plt, networkx as nx
from tqdm import tqdm
import numpy as np
n = 64
graph = nx.grid_2d_graph(n, n, periodic = True)

print('hello')
# t = 2 / (np.log(1 + np.sqrt(2)))
t = 2
model = Ising(graph = graph, temperature = t, doBurnin = False, mode = 'async')

fig, ax = plt.subplots()
data = ax.imshow(model.states.reshape(n, n))
N = 10000
plt.show(block = False)
r = (\
    model.sampleNodes[model.mode](model.nodeIDs)\
    for i in range(N)\
        )
from time import sleep
plt.show(0)
for i in tqdm(r):
    data.set_data(model.updateState(i).reshape(n, n))
    # fig.canvas.draw_idle()
    plt.pause(1e-300)
# plt.show()
