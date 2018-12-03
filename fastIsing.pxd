from models cimport Model
from cython cimport numeric
cimport numpy as np
cdef fused longdouble:
    long
    double
cdef class Ising(Model):
    cdef:
        public:
            object adj    # sparse adjacency matrix
            str magSide   # which side to sample on
            np.ndarray _H # external magnetic field
            double beta

    cdef double energy(self, \
                       int  node, \
                       long[:] index,\
                       double [:] weights,\
                       double nudge,\
                       long[:] states)


    # cdef _updateState(self, long [:] nodesToUpdate)

    cpdef long[::1] updateState(self, long[:] nodesToUpdate)
    cpdef np.ndarray burnin(self,\
                 int samples=*,\
                 double threshold =*)
    cpdef long[:, :] simulate(self, long samples)
