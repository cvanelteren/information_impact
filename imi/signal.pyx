#distutils: language = c++
import numpy as np
cimport numpy as np, cython
from cython.parallel cimport prange, threadid, parallel
from scipy.ndimage import gaussian_filter

from scipy import signal as ssignal

from plexsim.models.base cimport *
from plexsim.models.types cimport *
from cython.operator cimport dereference as deref

from libcpp.vector cimport vector
from libcpp.pair cimport pair

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
#


def matchT(Model m, object sig, double m0 = .5,
              size_t window = int(2e5), double pct = .01):

    from scipy import optimize
    cdef np.ndarray idx = np.array([])
    cdef size_t nmin = int(window * pct)

    cdef double tol = 2 * (1/ <double> m.nNodes)
    cdef double high, low

    while not idx.size:
        # setup model
        t = optimize.minimize(lambda x: abs(sig(x) - m0), 0.0)
#         m.states = 0
        m.t = t.x

        # generate buffer
#         buffer = m.simulate_mean(window)
        buffer = m.simulate(window).mean(1)
        idx = np.where(np.isclose(buffer, 0.5, atol = tol))[0]
        print(f"Setting m_0 ={m0}\tt={m.t}\t#tipps={idx.size}\tthreshold={nmin}")
        if idx.size > nmin:
            idx  = np.array([])
            high = m0 * 2
            low  = m0
            m0   = (high + low) / 2
            m0   = np.clip(m0, 0, 1)
            idx  = np.array([])
        else:
            m0 /= 2
    return t, idx, m0

from scipy import ndimage
cpdef np.ndarray detect_tipping(double[::1] signal,
                                double threshold,
                                double rtol = 0,
                                double atol = 0,
                                size_t spacing = 1,
                                size_t sigma = 0 ):
    cdef np.ndarray idx
    if sigma:
        signal = ndimage.median_filter(signal, sigma)
    # idx = np.where(np.abs(np.diff(np.sign(signal - threshold))), x = 1, y = 0)
    idx = np.where(np.isclose(signal, threshold, atol = atol, rtol = rtol))[0]
    idx =  idx[np.argwhere(np.diff(idx) > spacing)]
    return idx

# cpdef tuple find_tipping_inline(Model, m,
#                                 dict settings):
#     cdef np.ndarray bins = settings.get('bins', np.linspace(0, 1, 10))
#     cdef dict dist = {b : {} for b in bins}

#     # spawn models
#     cdef size_t threads = settings.get('n_jobs', mp.cpu_count() - 1)
#     cdef SpawnVec models = m._spawn(threads = threads)

#     # look around tipping point
#     cdef size_t neighborhood = settings.get("surround", 1000)
#     cdef size_t n_samples = settings.get('n_samples', 100)

#     cdef float threshold = 1/<float>(m.nNodes) 
#     cdef float tipping = np.mean(m.agentStates)
#     cdef bint tipping_found = False

#     cdef size_t tid
#     cdef Model tid_model
#     cdef state_t[::1] tid_state 
#     for sample in prange(n_samples):
#         # acquire model
#         tid = thread_id()
#         tid_model = models[tid]
#         # find tipping
#         tipping_found = False
#         while tipping_found != True:
#             tid_state = (<Model> tid_model.ptr)._updateState((<Model>tid_model.ptr)._sampleNodes(1))
#             if abs( sum(tipping - tid_state) ) <=  threshold:
#                tipping_found = True 
#         with gil:
#             dist[0] = dist.get(0, 0) + 1
#             # find state in the neighborhood
#             tmp = (<Model> tid_model.ptr).simulate(neighbood)
#             for state in tmp[1:]:
#                 idx = np.digitize(state.mean(), bins)
#                 bin_ = bins[idx]
#                 dist[bin_] = dist.get(bin_, 0) + 1
#     return dist 

cdef void move_buffer(vector[vector[state_t]] &buffer) nogil:
    # reshuffle the buffer around
    cdef size_t t
    for t in range(1, buffer.size()):
        buffer[t - 1] = buffer[t]

cdef double get_mean(vector[state_t] &buffer) nogil:
    cdef double avg = 0
    cdef size_t N = buffer.size()
    cdef size_t idx
    for idx in range(N):
        avg += buffer[idx]
    avg = avg / (<double> (N))
    return avg

cpdef bin_data(vector[vector[state_t]] &buffer, dict snapshots, double Z):
    # bin snapshots
    cdef size_t n = buffer.size()
    cdef tuple tmp
    for t in range(n):
        tmp = tuple(buffer[t])
        snapshots[tmp] = snapshots.get(tmp, 0) + Z

cdef bint check_threshold(vector[state_t] &buffer,
                         double threshold,
                         double allowance) nogil:
    cdef double lower = threshold - allowance
    cdef double upper = threshold + allowance
    if  lower  <= get_mean(buffer)  <= upper:
        return True
    return False

cpdef vector[size_t] detect_peaks(Model m,
                         size_t N,
                         double threshold,
                        double allowance, size_t burnin = 0) nogil:

    cdef size_t idx
    cdef vector[size_t] peaks
    m._simulate(burnin)

    for idx in range(N):
        m._updateState(m._sampleNodes(1)[0])
        if check_threshold(deref(m._states),
                           threshold,
                           allowance):
            peaks.push_back(idx)
    return peaks


cdef void equilibrate(Model m, double[::1] &p, size_t n_equilibrate = int(1e3)) nogil:
    with gil:
        m.reset()
    m._simulate(n_equilibrate)


cpdef tuple wait_tipping(Model m,
                         double[::1] bins,
                         size_t n_window  = 500,
                         size_t n_tipping = 100,
                         double allowance = 1.0,
                         size_t n_equilibrate = int(1e6)
                         ):
    #generate snapshots from tipping points

    # assume using potts based model
    allowance =  allowance/(<double> (m.adj._nNodes))
    if m.adj._nNodes % 2 == 0 and m.sampleSize == 1:
        allowance = 0
        print("Even number of nodes detected")
        print(f"Setting {allowance=}")

    # setup parameters
    import multiprocessing as mp
    cdef:
        size_t cpus = mp.cpu_count()
        double threshold = np.mean(m.agentStates)
        vector[vector[vector[state_t]]] buffer = np.zeros((cpus, n_window, m.adj._nNodes))
        double Z = 1/(<double>(n_tipping))
        dict snapshots = {} # output
        size_t peaks, ti, node, tid, num_tip = 0
        double[::1] p = (np.ones(m.nStates) * 1 / (<double>(m.nStates))).cumsum()

    print(f"Looking for tipping with {threshold=} and {allowance=}")
    cdef size_t[::1] tip_counter = np.ones(cpus, dtype = np.uintp) * 3

    cdef size_t[::1] loop_counter = np.zeros(cpus, dtype = np.uintp)
    cdef size_t counter = 0

    # start runs
    cdef unordered_map[size_t, vector[size_t]] tips
    cdef SpawnVec models = m._spawn(cpus)

    print("Equilibrating models")
    cdef size_t R = 10000
    cdef size_t[::1] r_counter = np.zeros(cpus, dtype = np.uintp)
    cdef vector[vector[vector[node_id_t]]] r = np.zeros((cpus, R, m.sampleSize), dtype = np.uintp)
    cdef vector[state_t] tmp_state

    # for tid in prange(cpus, nogil = True):
        # equilibrate((<Model> models[tid].ptr), p, n_equilibrate)
        # r[tid] = (<Model> models[tid].ptr)._sampleNodes(R)


    print("Starting to find tipping points")
    for num_tip in prange(n_tipping, nogil = True):
        tid = threadid()
        equilibrate((<Model> models[tid].ptr), p, n_equilibrate)
        r[tid] = (<Model> models[tid].ptr)._sampleNodes(R)
        # reset an equilibrate
        # for allowance +- 1/nNodes this is the max
        loop_counter[tid] = 0
        tip_counter[tid] = 0
        while tip_counter[tid] < 1:
            # update state
            move_buffer(buffer[tid])
            if r_counter[tid] == R:
                r[tid] = (<Model> models[tid].ptr)._sampleNodes(R)
                r_counter[tid] = 0
            buffer[tid][n_window - 1] = (<Model> models[tid].ptr)._updateState(r[tid][r_counter[tid]])
            loop_counter[tid] += 1
            r_counter[tid] += 1
            if check_threshold(buffer[tid][n_window - 1], threshold, allowance):
                # peaks = detect_peaks(buffer, threshold, allowance)
                # print(peaks, end = "\r")

                tips[tid].push_back(loop_counter[tid])
                tip_counter[tid] += 1
                with gil:
                    bin_data(buffer[tid], snapshots, Z)
                    counter = counter + 1
                    # print(f"Completed {counter=} out of {n_tipping=}", end = "\n")

            # print(f"Found {num_tip}", end = "\r")
    return snapshots, tips


cpdef tuple find_tipping(Model m,
                   double[::1] bins,
                   size_t n_samples,
                   size_t window        = 100,
                   size_t buffer_size   = int(1e4),
                   size_t sigma         = 100,
                   double rtol          = 0.,
                   double atol          = .01,
                   bint asymmetric      = False,
                   size_t spacing       = 1,
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
        size_t jdx, zdx
        size_t ni
        size_t tid
        SpawnVec models = m._spawn()
    # start
    #
    # cdef size_t n = n_samples
    # for ni in prange(n_samples, nogil = True):
    cdef vector[size_t] isi = []
    cdef size_t last_flip = 0
    cdef size_t count = 0
    for ni in range(n_samples):
        m.states = m.agentStates[0]
        tid = threadid()
        buffer_          = (<Model> models[tid].ptr).simulate(buffer_size)
        idx = detect_tipping(buffer_.mean(1), threshold = tipping_point,
                             atol = atol, rtol = rtol, spacing = spacing,
                             sigma = sigma)
        if len(idx):
            if idx.max() + window > buffer_.shape[0]:
                buffer_ = np.concatenate((buffer_,  m.simulate(window)))

        # bin around tipping points
        for zdx in range(len(idx)):
            jdx = idx[zdx]
            # only count diffs
            if zdx > 0:
                isi.push_back(idx[zdx] + idx[zdx - 1])
            if jdx - window >= 0:
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
    return dist, isi

# cdef _tipping(Model m,
              # size_t buffer_size,
              # )
