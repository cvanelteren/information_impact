# distutils: language=c++
from libcpp.vector cimport vector
import numpy as np
cimport numpy as np
import networkx as nx
import ctypes
from libc.stdio cimport printf

graph = nx.path_graph(3, nx.DiGraph())


a = {i: list(range(5)) for i in range(3)}

from libcpp.vector cimport vector
from cython.operator cimport preincrement, dereference
from libcpp.map cimport map

cdef:
    map[int, vector[int]] m = a
    map[int, vector[int]].iterator start = m.begin()
    map[int, vector[int]].iterator end   = m.end()
    int total = 0
with :
    while start != end:
        total += derefence(start).second.size()
        preincrement(start)
print(total)



