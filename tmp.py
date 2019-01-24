import networkx as nx
from Toolbox import infcy
from Models import fastIsing

g = nx.path_graph(10)

model = fastIsing.Ising(g)
nSamples = 100
while True:
    infcy.getSnapShots(model, nSamples)
