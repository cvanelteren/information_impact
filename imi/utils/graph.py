import networkx as nx
import numpy as np
__author__ = 'Casper van Elteren'
__email__ = "caspervanelteren@gmail.com"

# nx.draw(gc, pos = nx.circular_layout(gc, scale = 1e-5),)
def nx_layout(graph, layout = None):
    from datashader.bundling import hammer_bundle
    import pandas as pd
    if not layout:
        layout = nx.circular_layout(graph)
    data = [[node]+layout[node].tolist() for node in graph.nodes]

    nodes = pd.DataFrame(data, columns=['id', 'x', 'y'])
    nodes.set_index('id', inplace=True)

    edges = pd.DataFrame(list(graph.edges), columns=['source', 'target'])
    return nodes, edges, hammer_bundle(nodes, edges)


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

import random
class ConnectedSimpleGraphs:
    
    def __init__(self):
        """"
        Class to hold connected graphs of size n
        """
        self.graphs = {2 : [nx.path_graph(2)]}
        self.gm = nx.algorithms.isomorphism.GraphMatcher
        
    def generate(self, n):
        self.graphs = dict(sorted(self.graphs.items(), key = lambda x : x[0]))
        # get largest key already computed
        start = list(self.graphs.keys())[-1]
        while start < n:
            for base in self.graphs.get(start, []):
                # TODO add check for each key if all graphs are found
                for k in range(1, start + 1):
                    graph = self.__call__(base, k)
            start += 1
        return self.graphs
            
    def __call__(self, base, k : int):
        import itertools
        # generate new connected graph
        n = len(base) + 1 
        for nodes in itertools.permutations(list(base.nodes()), k):
            proposal = base.copy()
            add = True
            for node in nodes:
                proposal.add_edge(node, n)
            
            for gprime in self.graphs.get(n, []):
                if self.gm(gprime, proposal).is_isomorphic():
                    add = False
                    break
            if add:
                self.graphs[n] = self.graphs.get(n, []) + [proposal]
        return proposal
    
    def rvs(self, n, sparseness = None):
        """
        Generate random connected graph of size n
        """
        if not sparseness:
            sparseness = lambda : random.uniform(0, 1)
        # start from the same base
        proposal = self.graphs[2][0].copy()
        for ni in range(2, n):
            print(sparseness())
            k = int(sparseness() * ni)
            k = max((k, 1))
            for node in random.choices(list(proposal.nodes()), k = k):
                proposal.add_edge(ni, node)
        return proposal
            
