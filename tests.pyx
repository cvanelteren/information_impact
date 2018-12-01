# distutils: language=c
from cython import parallel
from libc.stdio cimport printf
def test_func():
    cdef int thread_id = -1
    with nogil, parallel.parallel(num_threads=10):
        thread_id = parallel.threadid()
        printf("Thread ID: %d\n", thread_id)
test_func()
