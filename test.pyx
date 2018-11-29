from cython.parallel cimport parallel, prange


cdef int i, j, k = 0
cdef int N = int(1e3)

for i in prange(0, N, nogil = True, num_threads = 4):
    for j in range(0, N):
        k += i + j
