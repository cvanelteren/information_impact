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
        vector[long]  _states
        vector[long]  _newstates # alias
        vector[long]   _nodeids
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

    cpdef vector[long] updateState(self,vector[long] nodesToUpdate)
    cdef vector[long] _updateState(self, vector[long]  nodesToUpdate) nogil


    cdef unordered_map[int, vector[long]]  sampleNodes(self, long Samples) nogil

    cdef unordered_map[int, vector[long]]  c_sample(self,
                    vector[long]  nodeIDs, \
                    int length,long nSamples,\
                    long long int  sampleSize,\
                    ) nogil

    cpdef simulate(self, long long int  samples)

    # cpdef long[::1] updateState(self, long[::1] nodesToUpdate)


    cpdef void reset(self)
    # cdef long[::1] updateState(self, int[:] nodesToUpdate)
