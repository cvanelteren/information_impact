cimport numpy as np
from libcpp.vector cimport vector
from libcpp.map cimport map
from libcpp.unordered_map cimport unordered_map
from sampler cimport Sampler
cdef struct Connection:
    vector[int] neighbors
    vector[double] weights
cdef class Model:
    cdef:
        # public
        # np.ndarray _states
        # np.ndarray  _nodeids
        # np.ndarray  agentStates
        long[::1] _states
        long[::1] _newstates # alias
        long[::1]  _nodeids
        long[::1]  agentStates
        # public sparse.csr.csr_matrix adj
        int _nNodes
        str __updateType
        str __nudgeType
        double[::1] __nudges

        Sampler sampler

        unordered_map[long, Connection] adj # adjacency lists

        # map[int, vector[int]] neighbors
        # map[int, vector[double]] weights
        # vector[int, vector[vector[int], vector[double]]] adj
        int _nStates
        #private
        dict __dict__
    cpdef void construct(self, object graph, \
                    list agentStates)

    cpdef long[::1] updateState(self, long[::1] nodesToUpdate)
    cdef long[::1]  _updateState(self, long[::1] nodesToUpdate)


    cdef  long[:, ::1] sampleNodes(self, long Samples)

    cdef  long[:, ::1] c_sample(self,
                    long[::1] nodeIDs, \
                    int length,long nSamples,\
                    long long int  sampleSize,\
                    )

    cpdef simulate(self, long long int  samples)

    # cpdef long[::1] updateState(self, long[::1] nodesToUpdate)


    cpdef void reset(self)
    # cdef long[::1] updateState(self, int[:] nodesToUpdate)
