# # distutils: language=c++
# from libcpp.vector cimport vector
# import numpy as np
# cimport numpy as np
# import networkx as nx
# import ctypes
# from libc.stdio cimport printf
# import time
#
# graph = nx.path_graph(3, nx.DiGraph())
# from fastIsing cimport Ising
# from cython cimport parallel
# a = {i: list(range(20)) for i in range(300)}
# from libcpp.vector cimport vector
# from cython.operator cimport preincrement, dereference
# from libcpp.map cimport map
# import multiprocessing as mp
# def func(idx):
#     global m, start
#     cdef int i, j
#     i = dereference(start).first + idx
#     l = dereference(start).second.size()
#     out = 0
#     for j in range(l):
#         out += m[i][j]
#     return out
# cdef map[int, vector[int]] m = a
# cdef map[int, vector[int]].iterator start = m.begin()
# cdef map[int, vector[int]].iterator end   = m.end()
# cdef long long total = 0
# cdef int j = 0
# cdef int k, l
# s = time.process_time()
# cdef int ss = dereference(start).first
# cdef int ee = dereference(end).first
#
# with nogil:
#     for j in parallel.prange(ss, ee, schedule = 'static'):
#         k    = dereference(start).first + j
#         l    = dereference(start).second.size()
#         for  j in range(l):
#             total = total + m[k][j]
#
# print(time.process_time() - s)
# print(total)
# s  = time.process_time()
#         #total += dereference(k).second.size()
# with nogil:
#     total = 0
#     while start != end:
#         for j in range(dereference(start).second.size()):
#              total += dereference(start).second[j]
#         preincrement(start)
# print(time.process_time() - s)
# with mp.Pool(4) as p:
#     p.map(func, range(ss, ee))
#
# print(total)
# s = time.process_time()
# t = 0
# for k, v in a.items():
#     t += sum(v)
#
# print(time.process_time() - s)
# print(t)
