from plexsim.models cimport *
import numpy as np
cimport numpy as np

from scipy.ndimage import gaussian_filter

cpdef build_dist(state_t[:, ::1] x, dict dist):
    cdef:
        size_t n = x.base.shape[0]
        size_t m = x.base.shape[1]

        tuple tmp
        double z = 1/<double>n
    for ni in range(n):
        tmp = tuple(x.base[ni])
        dist[tmp] = dist.get(tmp, 0) + 1
    return dist

cpdef find_tipping(Model m, \
                   size_t n = 100,\
                   size_t buffer_size = int(1e4),\
                   sigma = 100):

    # storage for sims
    cdef state_t[:, ::1] buffer_
    # getting the swap points
    cdef state_t[::1] swap
    # actual swap points
    cdef state_t[:, ::1] kronecker_deltas
    # building the dist
    cdef dict dist = {}

    # start
    cdef size_t ni = 0
    for ni in range(n):
        buffer_ = m.simulate(buffer_size)
        swap    = np.sign(gaussian_filter(np.mean(buffer_, axis = 1).round(), sigma))
        kronecker_deltas = buffer_.base[np.where(swap)]
        dist    = build_dist(kronecker_deltas, dist)
        # ni      = len(dist)

    cdef double v, z = sum(dist.values())
    cdef tuple k
    for k, v  in dist.items():
        dist[k] = v / z
    return dist
