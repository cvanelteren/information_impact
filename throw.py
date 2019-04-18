import matplotlib as mpl
from Models import fastIsing
import networkx as nx

g = nx.grid_2d_graph(10, 10)

m = fastIsing.Ising(g)