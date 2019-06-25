#distutils: language = c++
cimport cython
from Models.models cimport Model

cdef class Percolation(Model):
    cdef:
        double p

    cdef long[::1] updateState(self, long[::1] nodes) nogil