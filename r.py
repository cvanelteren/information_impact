from fastIsing import Ising
from models import Model
import networkx as nx, time
from numpy import *
from matplotlib.pyplot import *
graph = nx.barabasi_albert_graph(2000, 100)


m = Ising(graph, temperature = 1000)
m.mode='single'
m.magSide = 'pos'
s = time.time()
a = m.simulate(100000)


print(time.time() - s)
fig, ax = subplots()
ax.plot(a.mean(1))
show()
print(a.shape)
