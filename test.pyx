# distutils: language = c++
# distutils: extra_compile_args = -std=c++11

cdef extern from "<random>" namespace "std" nogil:
    cdef cppclass mt19937:
        mt19937() # we need to define this constructor to stack allocate classes in Cython
        mt19937(unsigned int seed) # not worrying about matching the exact int type for seed

    cdef cppclass uniform_real_distribution[T]:
        uniform_real_distribution()
        uniform_real_distribution(T a, T b)
        T operator()(mt19937 gen) # ignore the possibility of using other classes for "gen"


from posix.time cimport clock_gettime, timespec, CLOCK_REALTIME

cdef double test() nogil:
    global gen, seed, dist
    return dist(gen)

import numpy as np
cimport numpy as np
from libc.stdio cimport printf
cdef int i
cdef timespec ts
cdef int current
clock_gettime(CLOCK_REALTIME, &ts)
seed = ts.tv_sec

cdef:
    mt19937 gen = mt19937(seed)
    uniform_real_distribution[double] dist = uniform_real_distribution[double](0.0,1.0)
with nogil:
    for i in range(100):
        printf('%f %f \n', dist(gen), test())
