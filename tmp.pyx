# cimport cython
#
# from numpy cimport ndarray as ar
# from cython.parallel import prange
#
# @cython.boundscheck(False)
# @cython.wraparound(False)
# cpdef cysumpar(ar[double] A):
#   cdef double tot = 0
#   cdef int i, n = A.size
#   for i in prange(n, nogil = True):
#     tot += A[i]
#   return tot
