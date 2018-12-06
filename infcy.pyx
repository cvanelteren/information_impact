
# cython: infer_types=True
# distutils: language=c++
# __author__ = 'Casper van Elteren'
import numpy as np
cimport numpy as np
cimport cython

from cython.parallel cimport parallel, prange, threadid
# cimport numpy as np
# cimport cython
import IO, plotting as plotz, networkx as nx, functools, itertools, platform, pickle,\
 copy, time
# from pathos import multiprocessing as mp
import multiprocessing as mp
from models cimport Model
from fastIsing cimport Ising
# import multiprocess as mp
# import pathos.multiprocessing as mp
from tqdm import tqdm   #progress bar
#from joblib import Parallel, delayed, Memory
from libcpp.vector cimport vector
from libcpp.unordered_map cimport unordered_map
# from libcpp.concurrent_unordered_map cimport concurrent_unordered_map
from libc.stdio cimport printf
import ctypes
#
# # TODO: the numpy approach should be re-written in a dictionary only approach in order to prevent memory issues;
# # the general outline would be to yield the results and immediately bin them accordingly and write state to disk
# from models cimport Model
# # array = functools.partial(np.array, dtype = np.float16) # tmp hack
# from libc.stdio cimport printf
# cdef int _CORE = 1  # for imap
# INT16 = np.int16
def checkDistribution():
    '''Warning statement'''
    from platform import platform
    if 'windows' in platform().lower():
        print('Warning: Windows detected. Please remember to respect the GIL'\
              ' when using multi-core functions')
checkDistribution() # print it only once


# cdef int encodeState(long[::1] state) :
#     cdef int i, N = state.shape[0]
#     cdef int out = 1
#     for i in range(N):
#         if state[i] != -1:
#             out *=  2 ** i
#     return out
# def encodeState(state, nStates):
#     return int(''.join(format(1 if i == 1 else 0, f'0{nStates - 1}b') for i in state), 2)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cpdef int encodeState(long[::1] state):

    cdef int binNum = 1
    cdef int N = state.shape[0]
    cdef int i
    cdef int dec = 0
    for i in range(N):
        if state[i] == 1:
            dec += binNum
        binNum *= 2
    return dec

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cpdef vector[long] decodeState(int dec, int N):
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


import time
import faulthandler
faulthandler.enable()
from cython.operator cimport dereference as deref, preincrement as pre

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cpdef dict getSnapShots(Model model, int nSamples, int step = 1,\
                   int burninSamples = int(1e3)):
    # start sampling
    cdef unordered_map[int, double] snapshots
    # cdef dict snapshots = {}
    cdef int i
    cdef int N = nSamples * step
    cdef long[:, ::1] r = model.sampleNodes(N )
    cdef double Z = <double> nSamples
    cdef int idx
    cdef double past = time.process_time()
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

    cdef float past = time.process_time()
    # pre-declaration
    cdef double Z            = <double> repeats
    cdef double[:] copyNudge = model.nudges.copy()
    cdef bint reset          = True
    # loop stuff
    cdef long[:, ::1]  s     = np.array([decodeState(i, model._nNodes) for i in snapshots])
    cdef int N = s.shape[0]
    # cdef double[::1] out     = np.zeros(N * (deltas + 1) * model._nNodes * model._nStates)
    cdef double[:, :, :, ::1] out     = np.zeros((N , (deltas + 1),  model._nNodes,  model._nStates))
    cdef long[:, ::1] r      = model.sampleNodes(N * repeats * (deltas + 1) )
    cdef int k, delta, node, statei, counter, sc,  half = deltas // 2, n
    # pbar = tqdm(total = repeats)
    counter = 0
    sc      = 0

    cdef list kdxs        = list(snapshots.keys())
    cdef dict conditional = {}
    cdef long[::1] startState
    pbar = tqdm(total = N )
    cdef int idx, jdx
    cdef int tid = -1
    # cdef list models = [copy.deepcopy(prototype) for _  in range(mp.cpu_count())]
    # cdef Model model
    with nogil, parallel():
        # tid = threadid()
        # printf('%d ', tid)
        for n in prange(N, schedule = 'dynamic'):
            # tid   = threadid()
            # model = models[tid]
            with gil: # TODO: have a fix for this...
                for k in range(repeats):
                    for node in range(model._nNodes):
                        model._states[node] = s[n][node]
                        model._nudges[node] = copyNudge[node]
                    # reset simulation
                    reset   = True
                    for delta in range(deltas + 1):
                        # bin data
                        for node in range(model._nNodes):
                            for statei in range(model._nStates):
                                idx = (delta +  1) * (node + 1) * (statei + 1) + (n + 1)
                                if model._states[node] == model.agentStates[statei]:
                                    out[n, delta, node, statei] += 1 / Z
                        # update
                        # print(counter, sc, r.base.size, out.base.size)
                        jdx  = (delta +  1) * (node + 1) * (statei + 1) + (n + 1) * (k + 1)
                        # printf('%d ', jdx)
                        model._updateState(r[jdx])
                        # turn-off
                        if reset:
                            if model.__nudgeType == 'pulse' or \
                            model.__nudgeType    == 'constant' and delta >= half:
                                model._nudges[:] = 0
                                reset            = False
                pbar.update(1)
                conditional[kdxs[n]] = out.base[n]
    pbar.close()
    print(f"Delta = {time.process_time() - past}")
    return conditional
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
#     with mp.Pool(4) as p:
#         conditional = {kdx : res for kdx, res in zip(snapshots, p.imap(f, tqdm(models)))}
#     print(conditional)
#     print(f"Delta = {time.process_time() - past}")
#     return conditional


cpdef np.ndarray f(Worker x):
    # print('id', id(x))
    return x.parallWrap()


@cython.auto_pickle(True)
cdef class Worker:
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
                    if model.__nudgeType == 'pulse' or \
                    model.__nudgeType    == 'constant' and delta >= half:
                        model._nudges[:] = 0
                        reset            = False
            # pbar.update(1)
        # pbar.close()
        return out.base.reshape((deltas + 1, model._nNodes, model._nStates))
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
