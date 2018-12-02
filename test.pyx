# distutils: language=c++
from libcpp.vector cimport vector
import numpy as np
cimport numpy as np
import networkx as nx
import ctypes
from libc.stdio cimport printf

graph = nx.path_graph(3, nx.DiGraph())
adj   = nx.adj_matrix(graph).T
a     = [adj[:, n].data for n in range(graph.number_of_nodes())]

cdef vector[vector[int]] data
cdef vector[int] tmp
cdef vector[vector[int]] indices
cdef int N = graph.number_of_nodes()
cdef int i, j, k

data.resize(N)
indices.resize(N)
print(type(adj))
print(adj.todense())
for i in range(N):
    d = adj[i].data
    dd = adj[i].indices
    print(d, dd)
    for j in range(d.size):
        data[i].push_back(d[j])
        indices[i].push_back(dd[j])
print(data, indices)
