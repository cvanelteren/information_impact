# distutils: language=c++
# __author__ = 'Casper van Elteren'
from tqdm import tqdm  # progress bar
from cpython cimport PyObject, Py_XINCREF, Py_XDECREF
from cython.operator cimport dereference as deref, preincrement as prec
from timeit import default_timer as timer
from cython.view cimport array as cvarray
from libc.stdio cimport printf
from libcpp.unordered_map cimport unordered_map
from libc.stdlib cimport srand
from libcpp.string cimport string
from libcpp.vector cimport vector
from pyprind import ProgBar
from cpython.ref cimport PyObject
import copy
import multiprocessing as mp
from cython.parallel cimport parallel, prange, threadid
import time
import numpy as np
cimport numpy as np
cimport cython


cdef class Simulator:
    def __init__(self, Model m):
        self.model = m

        for idx, i in enumerate(m.agentStates):
            self.hist_map[i] = idx

    cpdef dict snapshots(self, size_t n_samples,
                         size_t step=2, int n_jobs = -1):
        cdef tuple tmp
        cdef dict snapshots = {}
        cdef:
            size_t nThreads = mp.cpu_count() - 1 if n_jobs == -1 else n_jobs
            SpawnVec models = self.model._spawn(nThreads)

            vector[vector[vector[node_id_t]]] r    = np.ndarray(
                            (nThreads,
                            step,
                            self.model.sampleSize),
                            dtype = np.uint)

            vector[vector[double]] states
            double Z = 1/<double>(n_samples)

        cdef size_t tid, s, i
        for i in prange(n_samples, nogil = True,
                        num_threads = nThreads,
                        schedule = "static"):
            tid = threadid()
            r[tid] = (<Model> models[tid].ptr)._sampleNodes(step)
            for s in range(step):
                (<Model> models[tid].ptr)._updateState(r[tid][s])
            with gil:
                tmp = tuple((<Model> models[tid].ptr).states)
                snapshots[tmp] = snapshots.get(tmp, 0) + Z
        
        #for i in range(n_samples):
        #     states = self.model.simulate(step + 1)[-1]
        #     tmp = tuple(states.base)
        #     snapshots[tmp] = snapshots.get(tmp, 0) + 1
        #     # self.model.reset()
        return snapshots

    # TODO: fix me; i have changed the way binning is done to arrays below
    # future me neew to swap out the dicts here with a c++ threadfreidnly access
    # cpdef dict running(self,
    #                    size_t n_samples,
    #                    size_t time_steps=10,
    #                    size_t steps=1,
    #                    bint center=False):
    #     """  Performs running  window over  :time_steps: for
    #     :n_samples: times separated with :steps:.

    #     :center:  controls whether  the buffer  is centered,
    #     Default is  that the last  time index is  the target
    #     state of the buffer. """

    #     # setup buffer
    #     cdef:
    #         tuple shape = (time_steps,
    #                                         self.model.adj._nNodes,
    #                                         self.model._nStates)
    #         double[:, :, ::1] bin_buffer = np.zeros(shape)
    #         state_t[:, ::1] buff = np.zeros(shape[:2], dtype=np.double)

    #     assert n_samples > time_steps, "time_steps needs to be bigger than number of samples"
    #     # setup targets
    #     cdef:
    #         state_t[::1] target
    #         size_t target_idx = time_steps // 2 if center else time_steps - 1

    #     cdef dict conditional = {}, snapshots = {}
    #     # start binning
    #     # init sample
    #     cdef size_t sample
    #     cdef double z = 1/<double > n_samples
    #     # init buffer
    #     buff = self.model.simulate(time_steps)
    #     for sample in range(n_samples):
    #         target = buff[target_idx]
    #         # obtain buffer or empty
    #         bin_buffer = conditional.get(tuple(target), np.zeros(shape))
    #         self.bin_data(buff, target, bin_buffer, 1)

    #         conditional[tuple(target)] = conditional.get(
    #             tuple(target), bin_buffer.base.copy())
    #         snapshots[tuple(target)] = snapshots.get(tuple(target), 0) + 1.

    #         # roll buffers
    #         buff = self.model.simulate(
    #             time_steps * steps)[time_steps * steps - time_steps:, :]

    #     return self.normalize(conditional, snapshots, True)

    cpdef dict normalize(self, dict conditional,
                         dict snapshots,
                         bint running=True):
        # normalize the probs
        cdef double z = 1 / <double > sum(snapshots.values())
        cdef double Z = Z
        cdef tuple k
        cdef np.ndarray v

        for k, v in conditional.items():
            Z = snapshots.get(k) * z
            if running:
                conditional[k] = v / snapshots.get(k)
            snapshots[k] = Z

        return dict(snapshots=snapshots,
                    conditional=conditional)

    cdef void bin_data(self,
                       state_t[:, ::1] &buff,
                       const state_t[::1] &target,
                       double[:, :, ::1] &bin_buffer,
                       double Z=1) nogil:

        # reset
        cdef size_t n = buff.shape[1]
        cdef size_t time_steps = buff.shape[0]
        cdef size_t idx, t

        # with gil:
        #     print(buff.shape,
        #           target.shape, bin_buffer.shape)

        # bin
        for t in range(time_steps):
            for node in range(n):
                idx = self.hist_map[buff[t, node]]
                bin_buffer[t, node, idx] += Z
        return

    cpdef dict forward(self,
                       dict snapshots,
                       size_t repeats=10,
                       np.ndarray time=np.arange(10),
                       int n_jobs = -1,
                       str schedule = "guided",
                       object chunksize = None,
                       ):

       from imi.utils.stats import check_allocation
       # setup buffers
       #
       cdef size_t cpus = mp.cpu_count()
       cdef:
          size_t time_steps = len(time)
          state_t[:, :, ::1] thread_state = np.zeros((cpus, time_steps,
                                                      self.model.adj._nNodes),
                                                      dtype=np.double)

          state_t[:, ::1] start_state = np.zeros((cpus, self.model.nNodes),
                                                 dtype=np.double)
          const state_t[:, ::1] states = np.array([i for i in snapshots.keys()],
                                            dtype=np.double).reshape(-1, self.model.nNodes)
          size_t nStates = len(snapshots)
          # spawn models parallely
          SpawnVec models = self.model._spawn(cpus)
          # shape of buffer
          tuple shape = (time_steps, self.model.nNodes, self.model.nStates)



       # cdef size_t type_bytes = np.uintp().nbytes
       # cdef size_t inner = repeats * time_steps
       # cdef size_t nbytes = inner * type_bytes
       # print("checking available bytes")
       # cdef double check = check_allocation(nbytes * cpus)

       # # memory reduction
       # if check < 1:
       #     print("Changed inner dimension")
       #     inner = <size_t>(inner * check)

       # print(f"setting inner dimension {inner}")

       # private variables
       cdef:
          # loop variables/shorthands
          size_t state_idx, trial, step, tid, node
          size_t NODES = self.model.adj._nNodes
          double Z = 1 / <double > (repeats)
          size_t T = <size_t > (np.max(time) + 1)

          # store only the following time points
          unordered_map[size_t, size_t] store_idx

       # setup indices to save
       for idx in range(len(time)):
           store_idx[time[idx]] = idx

       # prime state matrix
       # for idx, (k, v) in enumerate(snapshots.items()):
           # for ki in range(len(k)):
                # states[idx, ki] = k[ki]

       # cdef node_id_t[:, :, ::1] r = np.zeros((cpus, T, self.model.sampleSize), dtype = np.uintp)

       # cdef vector[size_t] thread_counter
       print(nStates)
       cdef:
           unordered_map[node_id_t, weight_t]  copyNudge = self.model.nudges
           size_t half = max((int(T * .5), 1))
           double[:, :, :, ::1] conditional = np.zeros((nStates, *shape)) # holder for probability decay
           vector[vector[vector[node_id_t]]] r = np.zeros((cpus, T, self.model.sampleSize), dtype=np.uintp)

       print("Starting parallel runs")
       for state_idx in prange(nStates,
                               nogil = True,
                               num_threads = cpus,
                               schedule = "static",
                               ):

           tid = threadid()
           # run trials
           for trial in range(repeats):
               # reset buffer
               thread_state[tid, 0, :] = states[state_idx]
               for node in range(NODES):
                   deref((< Model > models[tid].ptr)._states)[node] = states[state_idx, node]

               # reset nudges if exist
               for node in range(NODES):
                   if copyNudge.find(node) != copyNudge.end():
                           (<Model> models[tid].ptr)._nudges[node] = copyNudge[node]
               # # get random numbers
               r[tid] = (<Model> models[tid].ptr)._sampleNodes(T)
               # simulate a trace
               for step in range(1, T):
                   (< Model > models[tid].ptr)._updateState(r[tid][step])
                   # (< Model > models[tid].ptr)._updateState(
                       # (<Model> models[tid].ptr)._sampleNodes(1)[0])
                   # turn off nudge
                   if step >= half:
                       (<Model> models[tid].ptr)._nudges.clear()
                   # store data
                   if store_idx.find(step) != store_idx.end():
                       for node in range(NODES):
                         thread_state[tid, store_idx[step], node] = deref((<Model> models[tid].ptr)._states)[node]

               # bin buffer
               self.bin_data(thread_state[tid],
                            states[state_idx],
                            conditional[state_idx], Z)

       print("Done with trials")
       cdef dict python_condition = {}
       cdef size_t zdx
       for zdx in range(nStates):
            s = tuple(states.base[zdx])
            python_condition[s] = conditional.base[zdx].copy()
       return self.normalize(python_condition, snapshots, False)

       # for state_idx in prange(nStates, nogil=True, num_threads=cpus):
       #     tid = threadid()
       #     start_state[tid] = states[state_idx]

       #     with gil:
       #         # get or reset buffer
       #         bin_buffer.base[tid] = conditional.get(
       #             tuple(start_state[tid]), np.zeros(shape))
       #         # start_state[tid] =  np.array(next(states)).reshape(-1, (< Model > models[tid].ptr).nNodes)

       #     # start binning
       #     for trial in range(repeats):
       #         # reset buffer

       #         for node in range(NODES):
       #             (< Model > models[tid].ptr)._states[node] = start_state[tid, node]
       #             if copyNudge.find(node) != copyNudge.end():
       #                  (<Model> models[tid].ptr)._nudges[node] = copyNudge[node]
       #             thread_state[tid, 0, node] = start_state[tid, node]

       #         # with gil:
       #         #      r.base[tid] = (< Model > models[tid].ptr)._sampleNodes(T)

       #         r[tid] = (<Model> models[tid].ptr)._sampleNodes(T)
       #          # simulate a trace
       #         for step in range(1, T):
       #              # store data
       #             (< Model > models[tid].ptr)._updateState(r[tid, step])
       #             if step >= half:
       #                 (<Model> models[tid].ptr)._nudges.clear()

       #             if store_idx.find(step) != store_idx.end():
       #                 for node in range(NODES):
       #                      thread_state[tid, store_idx[step], node] = (< Model > models[tid].ptr)._states[node]

       #                  # thread_counter[tid] = 0

       #                  # thread_state[tid, store_idx[step]] = \
       #                  # (<Model> models[tid].ptr)._updateState((<Model> models[tid].ptr)._sampleNodes(1)[0])


       #                  # thread_state[tid, store_idx[step]] = \
       #                  # (<Model> models[tid].ptr)._updateState((<Model> models[tid].ptr)._sampleNodes(1)[0])

       #             # thread_counter[tid] += 1

       #         # bin buffer
       #         self.bin_data(thread_state[tid], start_state[tid], \
       #                           bin_buffer[tid], Z)
       #     with gil:
       #         conditional[tuple(start_state[tid])] = conditional.get(\
       #                      tuple(start_state[tid]),  bin_buffer.base[tid].copy())
       #         pbar.update(1)

       # return self.normalize(conditional, snapshots, False)



cpdef double kpn_entropy(double[:, ::1] x, size_t k, size_t p):
    """KPN entropy Lombardi et al.  (2016)"""
    assert p >= k, "P needs to be bigger or equal to p"
    assert x.ndim == 2, "x needs to be nsamples x nfeatures"
    cdef:
        size_t n = x.shape[0]
        size_t s = x.shape[1]


    from sklearn.neighbors import NearestNeighbors
    import scipy as sp

    # get nearest neighbors p, and k
    cdef double[:, ::1] pNN_distances, kNN_distances
    cdef size_t[:, ::1] pNN_indices

    cdef object NN = NearestNeighbors(n_neighbors = p, metric = "chebyshev")
    NN.fit(x)

    pNN_distances= NN.kneighbors()[0]
    pNN_indices  = np.asarray(NN.kneighbors()[1], dtype = np.uintp)

    cdef object multi_normal = sp.stats.multivariate_normal

    # start estimation
    cdef:
        double H = sp.special.digamma(n) - sp.special.digamma(k)
        double Gi, gxi, z = 1/<double> n
        # tmp storage
        # double[:, ::1] mean, cov
        # cdef size_t[::1] idx
        # integration range
        # double[::1] a = np.zeros(s)
        # double[::1] b = np.zeros(s)
    for ni in range(n):
        # reset buffers
        # a[:] = 0
        # b[:] = 0
        # get indices
        idx  = pNN_indices[ni]
        mean = x.base[idx].mean(0)
        cov  = np.cov(x.base[idx].T)
        gxi  = multi_normal.pdf(x.base[ni], mean = mean, cov = cov)

        a = x.base[ni] - pNN_distances[ni][k]
        b = x.base[ni] + pNN_distances[ni][k]

        # for jdx in range(s):
        #     a[jdx] = x[ni, jdx] - pNN_distances[ni][p - k - 1]
        #     b[jdx] = x[ni, jdx] + pNN_distances[ni][p - k - 1]

        Gi   = multi_normal.cdf(b,\
                               mean = mean, cov = cov,\
                               allow_singular = True) - \
                        multi_normal.cdf(a, mean = mean, cov = cov,\
                        allow_singular = True)
        H += z * (np.log(Gi) - np.log(gxi))
    return H



@cython.boundscheck(False) # compiler directive
@cython.wraparound(False) # compiler directive
cpdef mutualInformation(dict conditional, \
                          dict snapshots):
    '''
    Returns the node distribution and the mutual information decay
    Please note that it uses information implicitly from conditional to determine the correct dimensions
    of the data.
    '''
    cdef  np.ndarray px, H
    cdef int deltas, nodes, states
    for idx, (key, p) in enumerate(conditional.items()):
        if idx == 0:
            deltas, nodes, states = p.shape
            px = np.zeros((deltas, nodes, states))
            H  = np.zeros((deltas, nodes))
        # p    = np.asarray(p)
        H   -= entropy(p) * snapshots[key]
        px  += p  * snapshots[key] # update node distribution
    H += entropy(px)
    # H += np.nansum(px *  np.log2(px), -1)
    return px, H

cpdef entropy(np.ndarray p, ax = -1):
    """
    Computes entropy
    Removes negatives
    Expect the states to be at axis= - 1
    """
    return -np.nansum((p * np.log2(p)), axis = ax)

def KL(p1, p2):
    """
    KL-divergence between p1 and p2. P1 here is the true distribution and P2 is the proposal.
    """
    # p2[p2 == 0] = 1 # circumvent p = 0
    # p1[p1==0] = 1
    tmp = np.log(p1) - np.log(p2)
    kl = np.nansum( np.multiply(p1, tmp )\
                    , axis = -1)
    kl[np.isfinite(kl) == False] = 0 # remove x = 0
    return kl

cpdef rank_data(double[::1] &x):
    # prevent labeling non-uniques to the same bin
    cdef np.ndarray ranks = np.argsort(x)
    cdef np.ndarray output = np.zeros(len(x))
    cdef size_t idx, jdx
    for idx in range(x.size):
        for jdx in range(ranks.size):
            if x[idx] == x[ranks[jdx]]:
                output[idx] = jdx
                break
    return output

cpdef double permut_entropy(
    double[::1] &x, int t, int tau = 1, bint normalize = False):
    cdef size_t N = len(x)
    if t > N:
        raise "Window is larger than array"
    cdef dict dist = {}
    for idx in range(0, N - t, tau):
        tmp = tuple(np.argsort((x[idx : idx + t])))
        dist[tmp] = dist.get(tmp, 0) + 1

    cdef double z = sum(dist.values())
    dist = {k: v / z for k, v in dist.items()}
    cdef double H = entropy(np.array(list(dist.values())))
    if normalize:
        H /= np.log2(np.math.factorial(t))
    return H
