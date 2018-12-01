cdef class Model:
    cdef:
        public:
            long[::1] __states
            long[::1] __nodeids
            long[::1] agentStates
            double[::1] nudges
            # public sparse.csr.csr_matrix adj
            int __nNodes
            str __updateType
            str __nudgeType
            double[:] __nudges

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
