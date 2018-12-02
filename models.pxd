cdef class Model:
    cdef:
        public:
            long[::1] _states
            long[:] _nodeids
            long[:] agentStates
            # public sparse.csr.csr_matrix adj
            int _nNodes
            str __updateType
            str __nudgeType
            double[::1] __nudges
            dict neighbors
            dict weights

        #private
        dict __dict__


    cdef long [:, ::1] sampleNodes(self, long nSamples)
    cpdef void construct(self, object graph, \
                    list agentStates)

    cdef long[:, ::1] c_sample(self,
                    long[::1] nodeIDs, \
                    int length, int nSamples,\
                    int sampleSize,\
                    )
    # cdef long[::1] updateState(self, int[:] nodesToUpdate)
