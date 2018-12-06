# __author__ = 'Casper van Elteren'
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
        # np.ndarray _states
        # np.ndarray _newstates # alias
        # np.ndarray  _nodeids
        # np.ndarray  agentStates
        # public sparse.csr.csr_matrix adj
        int _nNodes
        str _updateType
        str __nudgeType
        double[::1] _nudges
        # np.ndarray _nudges

        Sampler sampler

        unordered_map[long, Connection] adj # adjacency lists

        # map[int, vector[int]] neighbors
        # map[int, vector[double]] weights
        # vector[int, vector[vector[int], vector[double]]] adj
        int _nStates
        #private
        dict __dict__ # allow dynamic python objects
    cpdef void construct(self, object graph, \
                    list agentStates)

    cpdef long[::1] updateState(self, long[::1] nodesToUpdate)
    cdef long[::1]  _updateState(self, long[::1] nodesToUpdate)


    cdef  long[:, ::1] sampleNodes(self, long Samples)


    cpdef simulate(self, long long int  samples)

    # cpdef long[::1] updateState(self, long[::1] nodesToUpdate)


    cpdef void reset(self)
