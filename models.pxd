cimport numpy as np

cdef class Model:
    cdef:
        # public
        # np.ndarray _states
        # np.ndarray  _nodeids
        # np.ndarray  agentStates
        long[::1] _states
        long[::1]  _nodeids
        long[::1]  agentStates
        # public sparse.csr.csr_matrix adj
        int _nNodes
        str __updateType
        str __nudgeType
        np.ndarray  __nudges
        dict neighbors
        dict weights
        int _nStates

        #private
        dict __dict__


    cdef long [:, ::1] sampleNodes(self, long nSamples)
    cpdef void construct(self, object graph, \
                    list agentStates)

    cdef long[:, ::1] c_sample(self,
                    long[::1] nodeIDs, \
                    int length, long long nSamples,\
                    int sampleSize,\
                    )
    cpdef void reset(self)
    # cdef long[::1] updateState(self, int[:] nodesToUpdate)
