# distutils: language = c++
# distutils: extra_compile_args = -std=c++11
# cython: infer_types=True
import numpy as np
cimport numpy as np
cimport cython

# @cython.final
# @cython.auto_pickle(True)
# cdef class Test:
#     cdef:
#         long[::1] _states
#         dict __dict__
#     def __init__(self, states):
#         self._states = states
#         print(self._states)
#
#     @property
#     def states(self): return np.asarray(self._states)
#
#     @states.setter
#     def states(self, value):
#         self._states [:] = value
#
#     cdef double func(self, long [::1] x):
#         cdef vector[long]
#         cdef int i, N = x.shape[0]
#         cdef double mu
#         for i in range(N):
#             mu += x[i] / <float> N
#         return mu

    # property _states:
    #     def __get__(self):
    #         return np.asarray(self._states)
    #     def __set__(self, values):
    #         cdef long[::1] self._states = self.states


from fastIsing cimport Ising
import networkx as nx, copy
from models cimport Model
from cpython.ref cimport PyObject
from libcpp.vector cimport vector
from libc.stdio cimport printf
# cdef Model tmp
cdef class Temp:
    cdef int a
    def __init__(self, a):
        self.a = a
    cdef double test(self):
        return 0.
g = nx.path_graph(3)
# cdef Ising tmp
def f():
    cdef vector[void *] vec
    cdef int i, n = 3
    # cdef Ising tmp
    cdef list ids = []
    cdef list classes  = [] # force reference counter?
    cdef Temp tmp
    for i in range(n):
        tmp = Temp(1)
        # classes.append(tmp)
        vec.push_back(<void *>tmp)
    print((<Temp>vec[0]).test())
    print(ids)
    # del classes
    print(ids)
f()
# print(l.states, k.states)
#
# print(np.unique(ids), ids)
