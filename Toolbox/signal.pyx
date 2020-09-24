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

    from scipy import signal as ssignal
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
    cdef long[::1] swap_
    for ni in range(n):
        buffer_ = m.simulate(buffer_size)
        # swap    = np.gradient(ssignal.detrend(gaussian_filter(np.mean(buffer_, axis = 1), sigma)))
        swap    = gaussian_filter(np.mean(mean, axis = 1))
        swap_    = np.where((abs(swap - .5).round(1)) == 0)[0]
        swap_    =  swap_.base[np.where(np.diff(np.insert(swap_, 0, -1)) > 1)[0] + 1]

        # swap    = (swap.base - swap.base.min()) / (swap.base.max() - swap.base.min())
        # swap_    = ssignal.find_peaks(np.abs(swap), \
        #                               height = .3 * swap.base.max(),\
                                      # distance = 2 * sigma)[0]

        # for idx in range(swap_.base.shape[0]):

        kronecker_deltas = buffer_.base[swap_]
        dist    = build_dist(kronecker_deltas, dist)
        # ni      = len(dist)

    cdef double v, z = sum(dist.values())
    cdef tuple k
    for k, v  in dist.items():
        dist[k] = v / z
    return dist
