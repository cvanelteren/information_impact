#distutils: language = c++
import numpy as np
cimport numpy as np

cimport cython
from cython.parallel cimport prange, threadid, parallel
from scipy.ndimage import gaussian_filter

from scipy import signal as ssignal

from plexsim.models cimport *

# cpdef build_dist(state_t[:, ::1] x, dict dist):
#     cdef:
#         size_t n = x.base.shape[0]
#         size_t m = x.base.shape[1]

#         tuple tmp
#         double z = 1/<double>n
#     for ni in range(n):
#         tmp = tuple(x.base[ni])
#         dist[tmp] = dist.get(tmp, 0) + 1
#     return dist

cpdef dict find_tipping(Model m,
                   double[::1] bins,
                   size_t n_samples,
                   size_t window      = 100,
                   size_t buffer_size = int(1e4),
                   size_t sigma       = 100,
                   double rtol    = 0.,
                   double atol    = .01,
                    bint asymmetric = False,
                    size_t spacing = 1,
                   double tipping_point = .5):

    # storage for sims
    # cdef state_t[:, ::1] buffer_
    # getting the swap points
    # cdef state_t[::1] swap
    # actual swap points
    # cdef state_t[:, ::1] kronecker_deltas
    # building the dist
    cdef dict dist = {b: {} for b in bins}
    cdef:
        size_t jdx
        size_t ni
        size_t tid
        SpawnVec models = m._spawn()
    # start
    #
    # cdef size_t n = n_samples
    # for ni in prange(n_samples, nogil = True):
    for ni in range(n_samples):
        m.states = m.agentStates[0]
        tid = threadid()
        # with gil:
        buffer_          = (<Model> models[tid].ptr).simulate(buffer_size)
        idx = np.where(np.isclose(buffer_.mean(1), tipping_point, atol = atol, rtol = rtol))[0]
        idx =  idx[np.argwhere(np.diff(idx) > spacing)]
        if len(idx):
            if idx.max() + window > buffer_.shape[0]:
                buffer_ = np.concatenate((buffer_,  m.simulate(window)))
        # bin around tipping points
        for jdx in idx:
            if jdx - window >= 0:
                # target = tuple(buffer_[jdx])
                # print(target)
                # dist[0][target] = dist[0].get(target, 0 ) + 1

                for state in buffer_[jdx - window : jdx + window]:
                    mag = tipping_point - state.mean()
                    if asymmetric:
                        mag = abs(mag)
                    b   = bins[np.digitize(mag, bins, right = False)]
                    dist[b][tuple(state)] = dist[b].get(tuple(state), 0 ) + 1
    # normalize counts
    for k, v  in dist.items():
        z = sum(v.values())
        for kk, vv in v.items():
            dist[k][kk] = vv / z
    return dist

# cdef _tipping(Model m,
              # size_t buffer_size,
              # )
