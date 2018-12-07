# # distutils: language=c++
# from libcpp.vector cimport vector
# import numpy as np
# cimport numpy as np
# # import networkx as nx
# # import ctypes
from libc.stdio cimport printf
# # import time
#
# from sampler cimport Sampler
#
# cdef Sampler s = Sampler(time.time(), 0, 1)
#
# from matplotlib.pyplot import subplots, show
cdef extern from "limits.h":
    int INT_MAX
    int RAND_MAX

from posix.time cimport clock_gettime, timespec, CLOCK_REALTIME

cdef timespec ts
cdef double current
clock_gettime(CLOCK_REALTIME, &ts)

# while  True:
#     current = ts.tv_sec + (ts.tv_nsec / 1000000000.)
#     print(current)
from libc.stdlib cimport rand, srand,  malloc, free, abort
cdef size_t shape = 3
from cython.view cimport array
from cython.parallel cimport parallel, prange, threadid
# cdef Py_ssize_t idx, i, n = 100
# cdef int * local_buf
# cdef size_t size = 10

import numpy as np
cimport numpy as np
cdef long[:, :] A = np.zeros((10, 10), dtype = int)
cdef unsigned int m = A.shape[0]
cdef unsigned int n = A.shape[1]
cdef long[:, :] tmp

cdef int j, N = 100
cdef long * car

cdef long * func(long * x, long n, long m) nogil:
    cdef int i, j
    for i in range(n):
        for j in range(m):
            x[i * j] = 1
    return x

import multiprocess as mp
cdef int tid
with nogil, parallel(num_threads = 4):
    tid = threadid()
    printf('%d ', tid)
    car = <long * > malloc(( m * n) * sizeof(long))
    with gil:
        test = np.zeros((100, 100))
        print(id(test), mp.current_process())
    car = func(car, m, n)
        # with gil:
        #     for i in range(n):
        #         for j in range(m):
        #             print(car[i * j])
    # tmp = <long[:m, :n]> car
    # test = local_buf

    # populate our local buffer in a sequential loop
    # for j in prange(N):
    #     pass

    free(car)
# n = int(1e5)
# x = np.zeros((n, 2))
#
# for i in range(n):
#     x[i] = [rand() / RAND_MAX, s.sample()]
# fig, ax = subplots()
# for i in range(2):
#     ax.hist(x[:, i], alpha = .5)
# show()
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
# with :
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
# with :
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
