import networkx as nx
__author__ = 'Casper van Elteren'
__email__  = "caspervanelteren@gmail.com"
def bfs_iso(graph, discovered, tree = nx.DiGraph()):
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
    return [bfs_iso(graph, {i:[None]}, nx.DiGraph()) for i in nodes]
