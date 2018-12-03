from models cimport Model
from cython cimport numeric
cimport numpy as np
# cdef fused longdouble:
    # long
    # double

cdef class Ising(Model):
    cdef:
        # public
        object adj    # sparse adjacency matrix
        str magSide   # which side to sample on
        np.ndarray _H # external magnetic field
        double beta

    cdef double energy(self, \
                       int  node, \
                       long[::1] index,\
                       double [::1] weights,\
                       double nudge,\
                       long[::1] states)


    # cdef _updateState(self, long [:] nodesToUpdate)

    cdef long[::1] _updateState(self, long[::1] nodesToUpdate)
    cpdef long[::1] updateState(self, long[::1] nodesToUpdate)
    cpdef np.ndarray burnin(self,\
                 int samples=*,\
                 double threshold =*)
    cpdef simulate(self, long samples)
