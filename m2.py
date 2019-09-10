import networkx as nx
from Models import FastIsing as FS



if __name__ == '__main__':
    for i in range(10):
        g = nx.path_graph(3)
        mi = FS.Ising(g)
        mi.matchMagnetization()
