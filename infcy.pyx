# cython: infer_types=True
# distutils: language=c++
# __author__ = 'Casper van Elteren'
import numpy as np
cimport numpy as np
cimport cython
import time
from cython.parallel cimport parallel, prange, threadid
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor
import copy
from cpython.ref cimport PyObject
# MODELS
from models cimport Model
from fastIsing cimport Ising

# progressbar
from tqdm import tqdm   #progress bar

# cython
from libcpp.vector cimport vector
from libc.stdlib cimport srand
from libcpp.unordered_map cimport unordered_map
from libc.stdio cimport printf
import ctypes
from cython.view cimport array as cvarray

# print gil stuff; no mp used currenlty so ....useless
def checkDistribution():
    '''Warning statement'''
    from platform import platform
    if 'windows' in platform().lower():
        print('Warning: Windows detected. Please remember to respect the GIL'\
              ' when using multi-core functions')
checkDistribution() # print it only once

# TODO rework this to include agentStates
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef int encodeState(long[::1] state) nogil:
    """Maps state to decimal number"""
    cdef:
        int binNum = 1
        int N = state.shape[0]
        int i
        int dec = 0
    for i in range(N):
        if state[i] == 1:
            dec += binNum
        binNum *= 2
    return dec

# TODO rework this to include agentStates
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef vector[long] decodeState(int dec, int N) nogil:
    """Decodes decimal number to state"""
    cdef:
        int i = 0
        # long[::1] buffer = np.zeros(N, dtype = int) - 1
        vector [long] buffer = vector[long](N, -1) # init with -1
    while dec > 0:
        if dec % 2:
            buffer[i] = 1
        i += 1
        dec = dec // 2
    return buffer

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cpdef dict getSnapShots(Model model, int nSamples, int step = 1,\
                   int burninSamples = int(1e3)):
    """
    Use single Markov chain to extract snapshots from the model
    """
    # start sampling
    cdef:
        unordered_map[int, double] snapshots
        int i
        int N          = nSamples * step
        long[:, ::1] r = model.sampleNodes(N )
        double Z       = <double> nSamples
        int idx
        double past    = time.process_time()
    pbar = tqdm(total = nSamples)
    for i in range(N):
        if i % step == 0:
            idx             = encodeState(model._states)
            snapshots[idx] += 1 / Z
            pbar.update(1)
        model._updateState(r[i])
    pbar.close()
    print(f'Found {len(snapshots)} states')
    print(f'Delta = {time.process_time() - past}')
    return snapshots

@cython.boundscheck(False) # compiler directive
@cython.wraparound(False) # compiler directive
@cython.nonecheck(False)
@cython.cdivision(True)
cpdef dict monteCarlo(\
               Model model, dict snapshots,
               int deltas = 10,  int repeats = 11,
               ):
    """
    Monte carlo sampling of the snapshots
    ISSUES:
        currently have to enforce the gil in order to not overwrite
        the model states. Best would be to copy the extensions. However,
        I dunno how to properly reference them in arrays
    """
    # TODO: solve the memory issues;  currently way too much is ofloaded to the memory
    # one idea would be to have the buffers inside the for loop. However prange is not possible
    # if that is used.
    cdef:
        float past = time.process_time()
    # pre-declaration
        double Z            = <double> repeats
        double[:] copyNudge = model.nudges.copy()
        bint reset          = True
    # loop stuff
    # extract startstates
        long[:, ::1] s = np.array([decodeState(i, model._nNodes) for i in snapshots])
        int states     = s.shape[0]

        # CANT do this inline which sucks either assign it below with gill or move this to proper c/c++
        # loop parameters
        int repeat, delta, node, statei, half = deltas // 2, state
        list kdxs        = list(snapshots.keys()) # extra mapping idx
        dict conditional = {}
        long[::1] startState
        int jdx
        double[:, :, :, ::1] out     = np.zeros((states , (deltas + 1), model._nNodes, model._nStates))
        long[  :,       ::1] r       = model.sampleNodes(states * repeats * (deltas + 1) )
        # list m = []
        vector[PyObject *] models
        Model tmp
        list models_ignore    = []
        int nThread, nThreads = 3
        int nNodes = model._nNodes
        str nudgeType = model._nudgeType

    for nThread in range(nThreads):
        tmp = copy.deepcopy(model)
        models_ignore.append(tmp)
        models.push_back(<PyObject *> tmp)

    # for i in range(nThreads)):
    #     models.push_back
    pbar = tqdm(total = states) # init  progbar
    # for al the snapshots
    for state in range(states):
        # repeats n times
        for repeat in range(repeats):
            # reset the buffers to the start state
            for node in range(model._nNodes):
                model._states[node] = s[state][node]
                model._nudges[node] = copyNudge[node]
            # reset simulation
            reset   = True
            # sample for N times
            for delta in range(deltas + 1):
                # bin data
                for node in range(model._nNodes):
                    for statei in range(model._nStates):
                        if model._states[node] == model.agentStates[statei]:
                            out[state, delta, node, statei] += 1 / Z
                # update
                jdx  = (delta +  1)  + (state + 1) * (repeat + 1)
                model._updateState(r[jdx])
                # turn-off the nudges
                if reset:
                    # check for type of nudge
                    if nudgeType == 'pulse' or \
                    nudgeType    == 'constant' and delta >= half:
                        model._nudges[:] = 0
                        reset            = False
        pbar.update(1)
        conditional[kdxs[state]] = out.base[state]
    pbar.close()
    print(f"Delta = {time.process_time() - past}")
    return conditional


@cython.boundscheck(False) # compiler directive
@cython.wraparound(False) # compiler directive
cpdef mutualInformation(dict conditional, int deltas, \
                          dict snapshots, Model model):
    '''
    Returns the node distribution and the mutual information decay
    '''
    cdef  np.ndarray px = np.zeros((deltas + 1, model._nNodes, model._nStates))
    cdef  np.ndarray H     = np.zeros((deltas + 1, model._nNodes))
    for key, p in conditional.items():
        # p    = np.asarray(p)
        H   -= np.nansum(p * np.log2(p), -1) * snapshots[key]
        px  += p  * snapshots[key] # update node distribution
    H += np.nansum(px *  np.log2(px), -1)
    return px, -H

# # belongs to worker
# @cython.boundscheck(False) # compiler directive
# @cython.wraparound(False) # compiler directive
# @cython.nonecheck(False)
# @cython.cdivision(True)
# cpdef dict monteCarlo(\
#                Model model, dict snapshots,
#                int deltas = 10,  int repeats = 11,
#                ):
#
#     cdef float past = time.process_time()
#      # store nudges already there
#     cdef list models = []
#     cdef dict params
#     import copy
#     for startidx, val in snapshots.items():
#         params = dict(\
#                     model      = model,\
#                     repeats    = repeats,\
#                     deltas     = deltas,\
#                     idx        = startidx,\
#                     startState = np.asarray(decodeState(startidx, model._nNodes)),\
#                     )
#
#         models.append(Worker(**params))
#     # cdef np.ndarray s = np.array([decodeState(q, model._nNodes) for q in snapshots], ndmin = 2)
#     cdef dict conditional
#     # with mp.Pool(2) as p:
#     with ThreadPoolExecutor(2) as p:
#         conditional = {kdx : res for kdx, res in zip(snapshots, p.map(f, tqdm(models)))}
#     print(conditional)
#     print(f"Delta = {time.process_time() - past}")
#     return conditional


cpdef np.ndarray f(Worker x):
    # print('id', id(x))
    return x.parallWrap()


@cython.auto_pickle(True)
cdef class Worker:
    """
    This class was used to wrap the c classes for use with the multiprocessing toolbox.
    However, the performance decreased a lot. I think this may be a combination of the nogil
    sections. Regardless the 'single' threaded stuff above is plenty fast for me.
    Future me should look into dealing with the gil  and wrapping everything in  c arrays
    """
    cdef int deltas
    cdef int idx
    cdef int repeats
    cdef np.ndarray startState
    cdef Model model
    # cdef dict __dict__
    def __init__(self, *args, **kwargs):
        # for k, v in kwargs.items():
        #     setattr(self, k, v)

        self.deltas     = kwargs['deltas']
        self.model      = kwargs['model']
        self.repeats    = kwargs['repeats']
        self.startState = kwargs['startState']
        self.idx        = kwargs['idx']

    cpdef np.ndarray parallWrap(self):
        cdef long[::1] startState = self.startState
        # start unpacking
        cdef int deltas           = self.deltas
        cdef int repeats          = self.repeats
        # cdef long[::1] startState = startState
        cdef Model model          = self.model
        # pre-declaration
        cdef double[::1] out = np.zeros((deltas + 1) * model._nNodes * model._nStates)
        cdef double Z              = <double> repeats
        cdef double[:] copyNudge   = model._nudges.copy()
        cdef bint reset            = True
        # loop stuff
        cdef long[:, ::1] r
        cdef int k, delta, node, statei, counter, half = deltas // 2
        # pbar = tqdm(total = repeats)
        for k in range(repeats):
            for node in range(model._nNodes):
                model._states[node] = startState[node]
                model._nudges[node] = copyNudge[node]
            # reset simulation
            reset   = True
            counter = 0
            r       = model.sampleNodes(repeats * (deltas + 1))
            for delta in range(deltas + 1):
                # bin data
                for node in range(model._nNodes):
                    for statei in range(model._nStates):
                        if model._states[node] == model.agentStates[statei]:
                            out[counter] += 1 / Z
                        counter += 1
                # update
                model._updateState(r[counter])

                # turn-off
                if reset:
                    if model._nudgeType == 'pulse' or \
                    model._nudgeType    == 'constant' and delta >= half:
                        model._nudges[:] = 0
                        reset            = False
            # pbar.update(1)
        # pbar.close()
        return out.base.reshape((deltas + 1, model._nNodes, model._nStates))
