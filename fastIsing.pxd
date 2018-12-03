from models cimport Model
from cython cimport numeric
from libcpp.vector cimport vector
from libcpp.map cimport map
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
                       vector[int] & index,\
                       vector[double] & weights,\
                       double nudge,\
                       long[::1] states)


    # cdef _updateState(self, long [:] nodesToUpdate)

    cdef long[::1] _updateState(self, long[::1] nodesToUpdate)
    cpdef long[::1] updateState(self, long[::1] nodesToUpdate)
    cpdef dict getSnapShots(self, int nSamples, int step =*,\
                       int burninSamples =*)
    cpdef np.ndarray burnin(self,\
                 int samples=*,\
                 double threshold =*)
    cpdef simulate(self, long samples)
