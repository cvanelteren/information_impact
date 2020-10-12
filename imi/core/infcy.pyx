# cython: infer_types=True
# distutils: language=c++
# __author__ = 'Casper van Elteren'


import numpy as np
cimport numpy as np
cimport cython
import time
from cython.parallel cimport parallel, prange, threadid
import multiprocessing as mp
import copy
from cpython.ref cimport PyObject


from plexsim.models cimport *
# progressbar
from tqdm import tqdm   #progress bar

from pyprind import ProgBar
# cython
from libcpp.vector cimport vector
from libcpp.string cimport string
from libc.stdlib cimport srand
from libcpp.unordered_map cimport unordered_map
from libc.stdio cimport printf
import ctypes
from cython.view cimport array as cvarray
from timeit import default_timer as timer
from cython.operator cimport dereference as deref, preincrement as prec
from cpython cimport PyObject, Py_XINCREF, Py_XDECREF

# print gil stuff; no mp used currently so ....useless
def checkDistribution():
    '''Warning statement'''
    from platform import platform
    if 'windows' in platform().lower():
        print('Warning: Windows detected. Please remember to respect the GIL'\
              ' when using multi-core functions')
checkDistribution() # print it only once
 
cdef class Simulator:
    def __init__(self, Model m):
        self.model = m

        self.hist_map = {idx : i for idx, i in enumerate(m.agentStates)}

    cpdef dict snapshots(self, size_t n_samples,\
                         size_t step = 1):


        cdef state_t[::1] states
        # cdef double z = 1 # / <double> n_samples
        # cdef tuple tmp

        cdef tuple tmp
        # need to add has function to unordered_map to access
        # make the loop below fully nogil. 
        cdef dict snapshots = {}
        # cdef unordered_map[string, double] snapshots
        for i in range(n_samples):
            states = self.model.simulate(step + 1)[-1]
            tmp = tuple(states.base)
            snapshots[tmp] = snapshots.get(tmp, 0) + 1
        return snapshots

    cpdef dict running(self, \
                       size_t n_samples,
                       size_t time_steps = 10,
                       size_t steps      = 1,
                       bint center       = False):
        """
        Performs running window over :time_steps: for :n_samples: times separated with
        :steps:.
        :center: controls whether the buffer is centered, Default is that the last time index is the target state of the buffer.
        """

        # setup buffer
        cdef:
            tuple shape                  = (time_steps, \
                                            self.model.adj._nNodes,\
                                            self.model._nStates)
            double[:, :, ::1] bin_buffer = np.zeros(shape)
            state_t[:, ::1] buff       = np.zeros(shape[:2], dtype = np.double)

        assert n_samples > time_steps, "time_steps needs to be bigger than number of samples"
        # setup targets
        cdef:
            state_t[::1] target
            size_t target_idx =  time_steps // 2 if center else time_steps - 1

        cdef dict conditional = {}, snapshots = {}
        # start binning
        # init sample
        cdef size_t sample
        cdef double z = 1/<double> n_samples
        # init buffer
        buff = self.model.simulate(time_steps)
        for sample in range(n_samples):
            target = buff[target_idx]
            # obtain buffer or empty
            bin_buffer = conditional.get(tuple(target), np.zeros(shape))
            self.bin_data(buff, target, bin_buffer, time_steps, 1)

            conditional[tuple(target)] = conditional.get(tuple(target), bin_buffer.base.copy()) 
            snapshots[tuple(target)] = snapshots.get(tuple(target), 0) + 1.

            # roll buffers
            buff = self.model.simulate(time_steps * steps)[time_steps * steps - time_steps:, :]

        return self.normalize(conditional, snapshots, True)


    cpdef dict normalize(self, dict conditional, \
                         dict snapshots, \
                         bint running = True):
        # normalize the probs
        cdef double z = 1 / <double> sum(snapshots.values())
        cdef double Z = Z
        cdef tuple k
        cdef np.ndarray v

        for k, v in conditional.items():
            Z = snapshots.get(k) * z
            if running:
                conditional[k] = v /snapshots.get(k) 
            snapshots[k]   = Z

        return dict(snapshots = snapshots, \
                    conditional = conditional)


    cdef void bin_data(self, \
                       state_t[:, ::1] buff,\
                       state_t[::1] target,\
                       double[:, :, ::1] bin_buffer,\
                       size_t time_steps,\
                       double Z = 1) nogil:

        # reset
        cdef size_t idx
        # bin
        for t in range(time_steps):
            for node in range(self.model.adj._nNodes):
                with gil:
                    idx = self.hist_map[buff[t, node]]
                bin_buffer[t, node, idx] += Z
        return

    cpdef dict forward(self, \
                       dict snapshots,\
                       size_t repeats    = 10,\
                       size_t time_steps = 10,\
                       ):

       #setup buffers
       cdef:
          size_t cpus    = mp.cpu_count()

          state_t[:, :, ::1] thread_state = np.zeros((cpus, time_steps,\
                                                      self.model.adj._nNodes), \
                                                      dtype = np.double)

          state_t[:, ::1] start_state = np.zeros((cpus, self.model.nNodes), \
                                                 dtype = np.double)

          state_t[:, ::1] states = np.array([i for i in snapshots.keys()], \
                                            dtype = np.double)


          node_id_t[:, :, ::1] r = np.zeros((cpus, repeats * time_steps, self.model.sampleSize), dtype = np.uintp)
          size_t nStates = len(states)

          # spawn models parallely
          SpawnVec models = self.model._spawn(cpus)

          # shape of buffer
          tuple shape = (time_steps, self.model.nNodes, self.model.nStates)
          dict conditional = {}
       # private variables
       cdef:
          PyObject* ptr
          size_t state_idx, trial, step, tid, node, NODES = self.model.adj._nNodes
          tuple tuple_start_state
          double[:,:, :, ::1] bin_buffer = np.zeros((cpus, *shape))
          double Z = 1  / <double> (repeats)

       pbar = ProgBar(nStates)
       # for state_idx in prange(nStates, nogil = True):
       for state_idx in prange(nStates, nogil = True):
           tid              = threadid()
           start_state[tid] = states[state_idx]

           with gil:
               # get or reset buffer
               bin_buffer.base[tid] = conditional.get(tuple(start_state[tid]), np.zeros(shape))
               r.base[tid] = (<Model> models[tid].ptr)._sampleNodes(repeats * time_steps)

           # start binning
           for trial in range(repeats):
               # reset buffer
               for node in range(NODES):
                   (<Model> models[tid].ptr)._states[node] = start_state[tid, node]

                   thread_state[tid, 0, node] = start_state[tid, node]
               for step in range(1, time_steps):
                   thread_state[tid, step] = (<Model> models[tid].ptr)._updateState(r[tid, trial*step + step])

               # bin buffer
               self.bin_data(thread_state[tid], start_state[tid], \
                                 bin_buffer[tid], time_steps, Z)

               # with gil:
           with gil:
               conditional[tuple(start_state[tid])] = conditional.get(\
                            tuple(start_state[tid]),  bin_buffer.base[tid].copy())
               pbar.update(1)
       # print("sanity check 2")
       # return dict(conditional = conditional, snapshots = snapshots)
       return self.normalize(conditional, snapshots, False)
        

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

# # TODO rework this to include agentStates
# cpdef int encodeState(long[::1] state) :
#     """
#     Maps state to decimal number.

#     NOTE this only works for binary,\
#     needs to be extended to include larger than binary case
#     """
#     cdef:
#         int binNum = 1
#         int N = state.shape[0]
#         int i
#         int dec = 0
#     for i in range(N):
#         if state[i] == 1:
#             dec += binNum
#         binNum *= 2
#     return dec

# # # 
# TODO rework this to include agentStates
# cpdef vector[long] decodeState(int dec, int N) :
#     """
#     Decodes decimal number to state.

#     NOTE this only works for binary,\
#     needs to be extended to include larger than binary case
#     """
#     cdef:
#         int i = 0
#         # long[::1] buffer = np.zeros(N, dtype = int) - 1
#         vector [long] buffer = vector[long](N, -1) # init with -1
#     while dec > 0:
#         if dec % 2:
#             buffer[i] = 1
#         i += 1
#         dec = dec // 2
#     return buffer


# cpdef dict getSnapShots(Model model, int nSamples, \
#                         int steps = 1,\
#                         int burninSamples = int(1e3), \
#                         int nThreads = -1,\
#                         verbose = False):
#     """
#     Determines the state distribution of the :model: in parallel. The model is reset
#     to random state and simulated for :step: + :burninSamples: steps after which
#     a single sample is drawn and added to the output :snapshots:

#     Input:
#         :model: a model according to :Models.models:
#         :nSamples: number of state samples to draw
#         :step: number of steps between samples
#     Returns:
#         :snapshots: dict containing the idx of the state as keys, and probability as values
#     """
#     cdef:
#         # unordered_map[int, double] snapshots
#         # unordered_map[int, vector[int]] msnapshots
#         dict snapshots = {}
#         int step, sample
#         int N          = nSamples * steps
#         # long[:, ::1] r = model.sampleNodes(N)
#         double Z       = 1 / <double> nSamples
#         int idx # deprc?
#         unordered_map[int, vector[int]].iterator got
#         double past    = timer()
#         list modelsPy  = []
#         Model tmp
#         cdef int tid,

#     nThreads = mp.cpu_count() if nThreads == -1 else nThreads
#     # threadsafe model access; can be reduces to n_threads
#     # for sample in range(nThreads):
#         # tmp = copy.deepcopy(model)
#         # tmp.reset()
#         # TODO: remove this
#         # try:
#             # tmp.burnin(burninSamples)
#         # except:
#             # pass
#         # tmp.seed += sample # enforce different seeds
#         # modelsPy.append(tmp)
#         # models_.push_back(PyObjectHolder(<PyObject *> tmp))


#     cdef SpawnVec models_ = model._spawn(nThreads)
#     # init rng buffers
#     cdef int sampleSize = model._sampleSize # model.nNodes if model.updateType != 'single' else 1

#     cdef node_id_t[:, :, ::1] r    = np.ndarray((nThreads, steps, sampleSize), \
#                                            dtype = long)
#     # cdef cdef vector[vector[vector[int][sampleSize]][nTrial]][nThreads] r    = 0
#     # cdef long[:, :, ::1] r = np.ndarray((nThreds, steps, sampleSize), dtype = long)
#     cdef PyObject *modelptr
#     if verbose:
#         pbar = ProgBar(nSamples)
#     #pbar = tqdm(total = nSamples)
#     # pbar = pr.ProgBar(nSamples)
#     cdef tuple state
#     cdef int counter = 0
#     # for sample in range(nSamples):
#     # for sample in prange(nSamples, nogil = True, \
#                          # schedule = 'static', num_threads = nThreads):
#     for sample in range(nSamples):
#         tid      = threadid()
#         modelptr = models_[tid].ptr
#         r[tid] = (<Model> modelptr)._sampleNodes(steps)
#         # perform n steps
#         for step in range(steps):
#             (<Model> modelptr)._updateState(r[tid, step])
#         # with gil:
#         state = tuple((<Model> modelptr).states)
#         snapshots[state] = snapshots.get(state, 0) + Z 
#         if verbose:
#             pbar.update(1)
#     if verbose:
#         print('done')
#         print(f'Found {len(snapshots)} states')
#         print(f"Delta = {timer() - past: .2f} sec")
#     return snapshots



# cpdef dict monteCarlo(\
#                Model model, dict snapshots,\
#                int deltas = 10,  int repeats = 11,
#                int nThreads = -1):
#     """
#     Monte-Carlo methods for estimating p(s_i^{t+\delta} | S).
#     Input:
#         :model: using the basic framework from Models.Models
#         :snapshots: dict with keys containing the state idx and the value containing its probability
#         :deltas: amount of time steps to simulate
#         :repeats:  number of times to repeat the simulation;
#         :nThreads: number of threads to use (default = -1 : all)
#     Returns:
#         :conditional: dict containing the probabilities of nodes of time (number of nodes x deltas)
#         for each state from :snapshots:
#     """
#     # TODO: solve the memory issues;  currently way too much is ofloaded to the memory
#     # one idea would be to have the buffers inside the for loop. However prange is not possible
#     # if that is used.
#     # print("Decoding..")
#     # setup
#     cdef:
#         float past = timer()
#     # pre-declaration
#         double Z              = <double> repeats
#         unordered_map[node_id_t, nudge_t]  copyNudge = model.nudges
#         bint reset            = True
#         # loop stuff
#         # extract startstates
#         # list comprehension is slower than true loops cython
#         # long[:, ::1] s = np.array([decodeState(i, model.adj._nNodes) for i in tqdm(snapshots)])
#         state_t[:, ::1] s = np.array([i for i in snapshots])
#         # long[:, ::1] s   = np.array([msnapshots[i] for i in tqdm(snapshots)])
#         int states       = len(snapshots)
#         # vector[int] kdxs = list(snapshots.keys()) # extra mapping idx

        # CANT do this inline which sucks either assign it below with gill or move this to proper c/c++
        # loop parameters
        # int repeat, delta, node, statei, half = deltas // 2, state
    #     dict conditional = {}
    #     # unordered_map[int, double *] conditional
    #     state_t[::1] startState
    #     int jdx

    #     # long[  :,       ::1] r       = model.sampleNodes( states * (deltas) * repeats)
    #     # list m = []

    #     int nNodes = model.adj._nNodes, nStates = model._nStates
    #     state_t[::1] agentStates = np.asarray(model.agentStates)
    #     str nudgeType = model._nudgeType

    #     unordered_map[state_t, size_t] idxer = {state : idx for idx, state in enumerate(agentStates)}

    #     list modelsPy = []
    #     vector[PyObjectHolder] models_
    #     Model threadModel
    #     PyObject *modelptr

    # # setup thread count
    # nThreads = mp.cpu_count() if nThreads == -1 else nThreads
    # # setup output storage

    # # threadsafe model access; can be reduces to n_threads
    # # for state in range(nThreads):
    # for state in range(nThreads):
    #     threadModel = copy.deepcopy(model)
    #     threadModel.seed += state # enforce different seeds
    #     # print(threadModel.t)
    #     # print(threadModel.nudges.base)
    #     # modelsPy.append(threadModel)
    #     models_.push_back(PyObjectHolder(<PyObject *> threadModel))


    # cdef int sampleSize = model._sampleSize # model.nNodes if model.updateType != 'single' else 1

    # # pre-define rng matrix
    # # cdef long[:, ::1] r = model.sampleNodes(states * deltas * repeats)

    # # bin matrix
    # # TODO: check the output array; threading cant be performed here; susspicious of overwriting
    # cdef double[:, :, :, ::1] out = np.zeros((nThreads     , deltas, \
    #                                           model.adj._nNodes, model._nStates))
    # cdef int nTrial = deltas * repeats
    # cdef node_id_t[:, :, ::1] r = np.ndarray((nThreads, nTrial,\
    #                                           sampleSize), dtype = np.uintp)
    # # cdef vector[vector[vector[int][sampleSize]][nTrial]][nThreads] r = 0

    # #pbar = tqdm(total = states) # init  progbar
    # # pbar = pr.ProgBar(states)

    # cdef int tid  # init thread id

    # cdef double ZZ

    # cdef Model t
    # print('starting runs')
    # # cdef double ZZ
    # with nogil, parallel(num_threads = nThreads):
    #     for state in prange(states, schedule = 'static'):
    #         tid         = threadid()
    #         modelptr    = models_[tid].ptr
    #         out[tid]    = 0 # reset buffer
    #         r[tid]      = (<Model> modelptr)._sampleNodes(nTrial)
    #         for repeat in range(repeats):
    #             for node in range(nNodes):
    #                 (<Model> modelptr)._states[node] = s[state, node]
    #                 (<Model> modelptr)._nudges[node] = copyNudge[node]

    #             for delta in range(deltas):
    #                 for node in range(nNodes):
    #                     jdx = idxer[(<Model> modelptr)._states[node]]
    #                     out[tid, delta, node, jdx] += 1 / Z
    #                 jdx = (delta + 1) * (repeat + 1)#  * (state + 1)
    #                 (<Model> modelptr)._updateState(r[tid, jdx - 1])
    #                 if nudgeType == 'pulse' or \
    #                 nudgeType    == 'constant' and delta >= half:
    #                     (<Model> modelptr)._nudges.clear()
    #         # TODO: replace this with   a concurrent unordered_map
    #         with gil:
    #             # note: copy method is required otherwise zeros will appear
    #             conditional[tuple(s.base[state])] = out.base[tid].copy()
    #             #pbar.update(1)
    # # for idx, si in enumerate(out.base):
    # #     conditional[tuple(s.base[idx])] = si

    # #pbar.close()
    # #print(f"Delta = {timer() - past: .2f} sec")
    # return conditional


# cpdef runMC(Model model, dict snapshots, int deltas, int repeats, dict kwargs = {}):
#     """ wrapper to perform MC and MI"""
#     cdef:
#         dict conditional = monteCarlo(model = model, snapshots = snapshots,\
#                 deltas = deltas, repeats = repeats,\
#                         **kwargs)
#         np.ndarray px, mi
#     px, mi = mutualInformation(conditional, snapshots)
#     return conditional, px, mi


# # cpdef reverseMC(Model model, int nSteps = 1000, \
# #                 int window = 10,\
# #                 int nSamples = 1000,\
# #                 bint reverse = 0,\
# #                 bint center = 0) :
# #     # TODO: add sample size function
# #     assert nSteps > window, "Size of nSteps need to be bigger than window size"
# #     cdef:
# #         int step
# #         int window_time
# #         int node
# #         int stateIdx
# #         state_t[:, ::1] windowData = np.zeros((window, model.nNodes),\
# #                                            dtype = long)
# #         tuple target
# #         double[:,:, ::1] buffer = np.zeros (( window, model.adj._nNodes, model._nStates))
# #         dict cpx = {}
# #         dict mapper = {i : idx for idx, i in enumerate(model.agentStates)}
# #         dict snapshots = {}
# #         long targetIdx 

#     targetIdx = window // 2 if center else window  -1
#     for step in range(nSteps):
#         if step % window == 0 and step > window:
#             # obtain target
#             target = tuple(windowData[targetIdx])
#             # obtain buffer, new copy
#             # buffer = cpx.get( target , buffer.copy())
#             buffer = cpx.get( target , np.zeros((window, model.adj._nNodes, model._nStates)))
#             for window_time in range(window):
#                 for node in range(model.adj._nNodes):
#                     stateIdx = mapper[windowData[window_time, node]]
#                     buffer[window_time, node, stateIdx] += 1
#                     # update counters
#             cpx[target] = cpx.get(target, buffer)
#             snapshots[target] = snapshots.get(target, 0) + 1
#         windowData = model.simulate(window)
#         # windowData[step  % window, :] = model._updateState(model._sampleNodes(1)[0])
#     # normalize
#     z = sum(snapshots.values())
#     for k, v in cpx.items():
#         cpx[k] = np.asarray(v.base) / <double> snapshots.get(k) 
#         snapshots[k] /= <double> z
#     px, mi = mutualInformation(cpx, snapshots)
#     return snapshots, px, cpx, mi

# cpdef  dict doTrial(Model m,\
#                    int nTrials,\
#                    np.ndarray[double] nudgeSizes,\
#                    int repeats  = 1000,\
#                    int nSamples = 1000,\
#                    int deltas   = 10,\
#                    int center  = 1,\
#                    int reverse = 1):
     
#      cdef:
#          SpawnVec models_ = m._spawn(mp.cpu_count())
        
#      cdef:
#          str node
#          node_id_t idx
#          int trial
#          int tid
#          dict trialNudges = {}
#          dict trialNudge
#          PyObject* ptr
#          Model tmp
#          double nudgeSize_

#      cdef int nNudges = len(nudgeSizes)

#      cdef int N = nTrials *nNudges;
#      pbar = ProgBar(N)
#      # for trial in prange(nTrials * nNudges, nogil = 1, \
#                          # num_threads = models_.size(), schedule = 'static'):
#      print("Starting trials")
#      for trial in range(N):
#          tid = threadid()
#          ptr = models_[tid].ptr
#          pbar.update(1)
#          nudgeSize_ = nudgeSizes[trial // nTrials]
#          nudgeSize_, trialNudge = _doTrial(ptr, nSamples = nSamples,\
#                                           deltas = deltas, \
#                                           reverse = reverse,\
#                                           center = center,\
#                                           repeats = repeats,\
#                                           nudgeSize = nudgeSize_,\
#                                                                )
#          trialNudges[nudgeSize_] = trialNudges.get(nudgeSize_, {})
#          for k, v in trialNudge.items():
#              trialNudges[nudgeSize_][k] = trialNudges[nudgeSize_].get(k, []) + [v]
#      return trialNudges

# cdef tuple _doTrial(PyObject* ptr,\
#               int nSamples,\
#               int deltas, \
#               int reverse,\
#               int center,\
#               int repeats,\
#               double nudgeSize,\
#               ):

#     cdef Model m = (<Model> ptr)
#     cdef dict snapshots
#     m.nudges = { }
#     cdef dict trialNudge = {}
#     m.reset()
#     if reverse:
#         snapshots, px, cpx, mi = reverseMC(m,\
#                                               nSteps = nSamples,\
#                                               window = deltas,\
#                                               center = center)
#     else:
#         snapshots = getSnapShots(m,\
#                                    nSamples = nSamples)
#         cpx, px, mi = runMC(m, \
#                       snapshots = snapshots,\
#                       deltas = deltas,\
#                       repeats = repeats)

#     cpx_ = np.array([i for i in cpx.values()])
#     trialNudge["control"] = dict(snapshots = snapshots, cpx = cpx,\
#                                  px = px, mi = mi)
#     for node, idx in m.mapping.items():
#         m.nudges = {node : nudgeSize}
#         if reverse:
#             cpxn = reverseMC(m,\
#                                  nSteps = nSamples,\
#                                  window = deltas,\
#                                  center = center)[2]
#         else:
#             cpxn = runMC(m, \
#                             snapshots = snapshots,\
#                             deltas = deltas,\
#                             repeats = repeats)[0]
#         cpxn_ = np.zeros(cpx_.shape)
#         for jdx, (k, v) in enumerate(cpx.items()):
#             cpxn_[jdx] = cpxn.get(k, 0)
#             nudge_impact = KL(cpx_, cpxn_)
#         trialNudge[node] = trialNudge.get(node, nudge_impact) 
#     return nudgeSize, trialNudge

    
