# cython: infer_types=True
# distutils: language=c++
__author__ = 'Casper van Elteren'
import numpy as np
cimport numpy as np
cimport cython

from cython.parallel cimport parallel, prange, threadid
# cimport numpy as np
# cimport cython
import IO, plotting as plotz, networkx as nx, functools, itertools, platform, pickle,\
fastIsing, copy, time
# from pathos import multiprocessing as mp
import multiprocessing as mp
# import multiprocess as mp
# import pathos.multiprocessing as mp
from tqdm import tqdm   #progress bar
#from joblib import Parallel, delayed, Memory
import ctypes

# TODO: the numpy approach should be re-written in a dictionary only approach in order to prevent memory issues;
# the general outline would be to yield the results and immediately bin them accordingly and write state to disk
from models cimport Model
# array = functools.partial(np.array, dtype = np.float16) # tmp hack
from libc.stdio cimport printf
cdef int _CORE = 1  # for imap
INT16 = np.int16
def checkDistribution():
    '''Warning statement'''
    from platform import platform
    if 'windows' in platform().lower():
        print('Warning: Windows detected. Please remember to respect the GIL'\
              ' when using multi-core functions')
checkDistribution() # print it only once

@cython.boundscheck(False) # compiler directive
@cython.wraparound(False)
@cython.cdivision(True) # compiler directive
@cython.nonecheck(False)
cpdef dict getSnapShots(Model model, int nSamples, int step = 1,\
                   int burninSamples = int(1e3)):
    # start sampling
    cdef dict snapshots = {}
    cdef double past = time.process_time()


    cdef int i

    cdef long long N = nSamples * step
    cdef long[:, :] r = model.sampleNodes(N )
    cdef double Z = <double> nSamples
    pbar = tqdm(total = N)
    with nogil, parallel():
        for i in prange(N, schedule = 'static'):
            with gil:
                if i % step == 0:
                    state = tuple(model.states)
                    snapshots[state] = snapshots.get(state, 0) + 1 / Z
                # model.updateState(next(r))
                model.updateState(r[i])
                pbar.update(1)
    pbar.close()
    print(f'Found {len(snapshots)} states')
    print(f'Delta = {time.process_time() - past}')
    return snapshots



'''
The alt functions currently only probe the system and the nodes. The plan is to build from here.
Conditions can be added, however the problem resides in the fact that it assumes time-symmetry;
it only calculates the probability in forward time, assuming that this is the same for backwards time.
For undirected graphs the probability of transition forwards will be the same as backwards, as there is
no inherrent asymmetry in the information flow. The non-alt functions have the ability to measure this flow;
however it requires to keep track of the states, which yields memory issues. As a proof of principle
one can do this for small graphs, but larger graphs will need tons of ram.
'''
@cython.boundscheck(False) # compiler directive
@cython.wraparound(False) # compiler directive
@cython.nonecheck(False)
@cython.cdivision(True)
cpdef dict monteCarlo_alt(\
               Model model, dict snapshots,
               long deltas = 10,  long repeats = 11,
               ):
    cdef dict conditional = {}

    cdef double past = time.process_time()

    # map from node state to idx
    cdef dict sortr = {i : idx for idx, i in \
                       enumerate(model.agentStates)}

    cdef str nudgeMode = model.__nudgeType
    cdef double[:] copyNudge = model.__nudges.copy() # store nudges already there
    # for k in range(repeats):


    cdef long[:, :] s = np.array([q for q in snapshots])
    cdef int N = s.shape[0], n

    cdef double[ :, :, :, ::1] out = np.zeros(\
               (\
                N, deltas + 1,\
                 model._nNodes,\
                  model._nStates,\
                  ))
    pbar = tqdm(total = N)

    cdef long[:, ::1] r = model.sampleNodes(N * repeats * (deltas + 1))
    print('Starting loops')
    # loop declarations
    cdef double Z = <double> repeats
    cdef long[:] states = model._states
    cdef double[::1] nudges = model.__nudges
    cdef int k, delta, node, statei
    half  = deltas // 2
    cdef bint reset = True
    cdef long[::1] agentStates = model.agentStates
    cdef long long idx
    cdef long chunk = N // 4

    # with nogil, parallel():
        # for n in prange(N, schedule = 'static', chunksize = chunk):
    # for n in range(N):
    for n in range(N):
        for k in range(repeats):
            for node in range(model._nNodes):
                states[node] = s[n, node]
                nudges[node] = copyNudge[node]
            reset        = True

            for delta in range(deltas + 1):
                # bin data()

                # idx   = np.digitize(state, model.agentStates)
                # out[delta, idx] += 1 / repeats
                for node in range(model._nNodes):
                    for statei in range(model._nStates):
                        if states[node] == agentStates[statei]:
                            out[n, delta, node, statei] += 1 / Z
                    # model.updateState(model.sampleNodes[model.mode](nodeIDs))
                # model.updateState(model.sampleNodes(1)[0])
                # print(np.asarray(r[(n + 1) * (k + 1) * (delta + 1)]))
                idx = (n + 1) * (k + 1) * (delta + 1)
                # with gil:
                #     model.updateState(r[idx])

                model.updateState(r[idx])

                # check if pulse turn-off
                if reset:
                    if nudgeMode == 'pulse' or nudgeMode == 'constant' and delta >= half:
                        for node in range(model._nNodes):
                            nudges[node] = copyNudge[node]
                        reset =  False

        # with gil:
        conditional[tuple(s[n])] = out[n]
        pbar.update(1)
    pbar.close()
    print(f"Delta = {time.process_time() - past}")
    return conditional

# @cython.boundscheck(False) # compiler directive
# @cython.wraparound(False) # compiler directive
# cpdef dict parallelMonteCarlo_alt(\
#           long[:] startState, object model, int repeats,\
#           int deltas,\
#           ):
#     cdef dict conditional = {}
#     # output time x nodes x node states
#     cdef double[:, :, :] out = np.zeros(\
#                                    (\
#                                     deltas + 1,\
#                                      model._nNodes,\
#                                       model._nStates,\
#                                       ))
#
#     # map from node state to idx
#     cdef dict sortr = {i : idx for idx, i in \
#                        enumerate(model.agentStates)}
#
#     cdef str nudgeMode = model.__nudgeType
#     cdef double[:] copyNudge = model.__nudges.copy() # store nudges already there
#     # for k in range(repeats):
#     half  = deltas // 2
#     reset = False
#     cdef long[:, :] r = model.sampleNodes(repeats * (deltas + 1))
#     cdef int k, delta, node
#     cdef int counter = 0
#     for k in range(repeats):
#         # start from the same point
#         model.states[:] = startState.copy()
#         model.__nudges[:] = copyNudge.copy()
#         # print(model.nudges, model.states)
#         reset        = True
#
#         for delta in range(deltas + 1):
#             # bin data()
#             state = model._states
#             # idx   = np.digitize(state, model.agentStates)
#             # out[delta, idx] += 1 / repeats
#             for node in range(model._nNodes):
#                 out[delta, node, sortr[state[node]]] += 1 / repeats
#                 # model.updateState(model.sampleNodes[model.mode](nodeIDs))
#                 # model.updateState(model.sampleNodes(1)[0])
#             model.updateState(r[counter])
#             counter += 1
#             # check if pulse turn-off
#             if reset:
#                 if nudgeMode == 'pulse' or nudgeMode == 'constant' and delta >= half:
#                     model.__nudges[:] = copyNudge.copy()
#                     reset =  False
#     # print('here')
#     return {tuple(startState) : np.asarray(out)}

@cython.boundscheck(False) # compiler directive
@cython.wraparound(False) # compiler directive
cpdef mutualInformation_alt(dict conditional, int deltas, \
                          dict snapshots, Model model):
    '''
    Returns the node distribution and the mutual information decay
    '''
    px = np.zeros((deltas + 1, model._nNodes, model._nStates))
    H  = np.zeros((deltas + 1, model._nNodes))
    for key, p in conditional.items():
        p = np.asarray(p)
        H  += np.nansum(p * np.log2(p), -1) * snapshots[key]
        px += p * snapshots[key] # update node distribution
    H -= np.nansum(px * np.log2(px), -1)
    return px, H
