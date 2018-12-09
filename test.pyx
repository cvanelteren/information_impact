# distutils: language = c++
# distutils: extra_compile_args = -std=c++11
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
from models cimport Model
import networkx as nx, copy
g = nx.path_graph(3)
m = Ising(g, 2)
from cpython.ref cimport PyObject
cdef PyObject * ptr = <PyObject *> m
from libcpp.vector cimport vector
from cython.operator cimport dereference, preincrement
from libc.stdlib cimport malloc, free, abort
# cdef vector[Ising] test
# cdef class Test:
#     cdef void *ptr
#     def __init__(self, a):
#         if self.ptr is not NULL:
#             free(self.ptr)
#         self.a = a
#     @staticmethod
#     cdef create(void* ptr):
#         p = Test(self.a)
#         p.ptr = ptr
#         return p

# cdef Ising test = Ising(g, 1)
cdef long[:, ::1] func(long[:, ::1] x ):
    return x
from cython.view cimport array as cvar
cdef  long[10][10] test
func(test)
print(test)
