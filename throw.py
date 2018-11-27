from fastIsing import Ising
import networkx as nx
from numpy import *
from matplotlib.pyplot import *
graph = nx.path_graph(3)
model = Ising(graph = graph, temperature = 10)

#model.states.fill(1)
n = 100
xs = []
for t in linspace(0, 5, n):
    model.t =  t
    model.states.fill(1)
    x = abs(model.simulate(100).mean())
    xs.append(x)
plot(xs)

import test
test.f()