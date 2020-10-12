# get all definition including typedefs
from plexsim.models cimport *
from libcpp.unordered_map cimport unordered_map

# ctypedef long state_t 
cdef class Simulator:
    cdef Model model
    cdef unordered_map[state_t, size_t] hist_map

    cpdef dict snapshots(self, size_t n_samples, size_t step=*)

    cpdef dict running (self, \
                        size_t n_samples,\
                        size_t time_steps =*,\
                        size_t steps = *,\
                        bint center =*
                        )

    cdef void bin_data(self, state_t[:, ::1] buffer,\
                       state_t[::1] target,\
                       double[:, :, ::1] bin_buffer, \
                       size_t time_steps,\
                       double Z = *
    ) nogil

    cpdef dict forward(self,\
                      # size_t n_samples =*,\
                       dict snapshots,\
                      size_t repeats =*,\
                      size_t time_steps =*,\
                      )

    cpdef dict normalize(self,\
                         dict conditional,\
                         dict snapshots,\
                         bint running =*
                         )
    

