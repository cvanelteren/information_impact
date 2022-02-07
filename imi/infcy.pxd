# get all definition including typedefs
from plexsim.models.base cimport *
from plexsim.models.types cimport *
from libcpp.unordered_map cimport unordered_map

import numpy as np
cimport numpy as np

# ctypedef long state_t 
cdef class Simulator:
    cdef Model model
    cdef unordered_map[state_t, size_t] hist_map

    cpdef dict snapshots(self, size_t n_samples, size_t step=*,
                         int n_jobs =*)

    # cpdef dict running (self, \
    #                     size_t n_samples,\
    #                     size_t time_steps =*,\
    #                     size_t steps = *,\
    #                     bint center =*
    #                     )

    cdef void bin_data(self, state_t[:, ::1] &buffer,\
                       const state_t[::1] &target,\
                       double[:, :, ::1] &bin_buffer, \
                       double Z = *
    ) nogil

    cpdef dict forward(self,\
                      # size_t n_samples =*,\
                      dict snapshots,
                      size_t repeats =*,
                      np.ndarray time =*,
                      int n_jobs =*,
                      str schedule =*,
                      object chunksize =*
                      )

    cpdef dict normalize(self,\
                         dict conditional,\
                         dict snapshots,\
                         bint running =*
                         )
    

