import networkx as nx
import numpy as np
__author__ = 'Casper van Elteren'
__email__ = "caspervanelteren@gmail.com"


def bfs_iso(graph, discovered, tree=nx.DiGraph()):
    """
    Breadth first search isomorphism algorithm.
    Constructs a directed tree-like graph from a node outwards
    """
    d = {}
    for source, pred in discovered.items():
        for neighbor in graph.neighbors(source):
            # don't consider where you came from
            if neighbor not in pred:
                tree.add_edge(source, neighbor)
                # don't go to already discovered nodes
                if neighbor not in discovered.keys():
                    d[neighbor] = d.get(neighbor, []) + [source]

    if d:
        bfs_iso(graph, d, tree)
    # print(discovered, d)
    return tree


def construct_iso_tree(nodes, graph):
    return [bfs_iso(graph, {i: [None]}, nx.DiGraph()) for i in nodes]


def make_connected(g) -> nx.Graph:
    # obtain largest set
    largest = max(nx.connected_components(g), key=lambda x: len(x))
    while len(largest) != len(g):
        for c in nx.connected_components(g):
            for ci in c:
                if ci not in largest:
                    # maintain the degree but add a random edge
                    for neighbor in list(g.neighbors(ci)):
                        target = np.random.choice(list(largest))
                        g.remove_edge(ci, neighbor)
                        g.add_edge(target, ci)
                        largest.add(ci)
                    # check in case neighbor has no degree
                    if ci not in largest:
                       g.add_edge(ci, np.random.choice(list(largest)))
    return g


def powerlaw_graph(n, gamma=1, connected=False, base=nx.Graph):
    deg = np.arange(1, n) ** -float(gamma)
    deg = np.asarray(deg * n, dtype=int)
    deg[deg == 0] = 1
    if deg.sum() % 2:
        deg[np.random.randint(deg.size)] += 1
    g = nx.configuration_model(deg, base())
    if connected:
        g = make_connected(g)
    return g

def recursive_tree(r, jump = 0):
    g = nx.Graph()
    g.add_node(0)
    sources = [0]
    n = len(g)
    while r > 0:
        newsources = []
        for source in sources:
            for ri in range(r):
                n += 1
                g.add_edge(source, n)
                newsources.append(n)
#         print(newsources)
        r -= 2 + jump
        sources = newsources
    return g
