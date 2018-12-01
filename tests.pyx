# distutils: language=c
from cython cimport parallel
from libc.stdio cimport printf
cdef test_func():
    cdef long thread_id = -1
    cdef long i,k, j = 0, N = 10000
    with nogil, parallel.parallel():
        thread_id = parallel.threadid()
        printf("Thread ID: %d\n", thread_id)
        for i in parallel.prange(1, N):
            for k in range(1, N):
                j += k
        printf('%d', j)
test_func()
