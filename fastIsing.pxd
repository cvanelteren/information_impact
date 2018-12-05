from models cimport Model
from cython cimport numeric
from libcpp.vector cimport vector
from libcpp.map cimport map
from libcpp.unordered_map cimport unordered_map
cimport numpy as np
# cdef fused longdouble:
    # long
    # double
cdef extern from "vfastexp.h":
    double exp_approx "EXP" (double)


cdef struct Connection:
    vector[int] neighbors
    vector[double] weights




cdef class Ising(Model):
    cdef:
        # public
        long _magSide   # which side to sample on
        double[:] _H # external magnetic field
        double beta


    cdef double energy(self, \
                       int  node, \
                       long[::1] states) nogil


    # cdef _updateState(self, long [:] nodesToUpdate)
    # c binding
    cpdef long[::1] updateState(self, long[::1] nodesToUpdate)
    cdef long[::1] _updateState(self, long[::1] nodesToUpdate) nogil
    # # python wrapper
    # cpdef long[::1] updateState(self, long[::1] nodesToUpdate)


    cpdef np.ndarray[double] burnin(self,\
                 int samples=*,\
                 double threshold =*)
