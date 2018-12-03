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

# TODO: the numpy approach should be re-written in a dictionary only approach in order to prevent memory issues;
# the general outline would be to yield the results and immediately bin them accordingly and write state to disk

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
@cython.wraparound(False) # compiler directive
cpdef getSnapShots(object model, int nSamples, int step = 1,\
                   int parallel = mp.cpu_count(), int burninSamples = int(1e3)):
    # start sampling
    cdef dict snapshots = {}
    cdef double past = time.process_time()

    pbar = tqdm(total = nSamples)

    cdef int i
    cdef long[:, ::1] r = model.sampleNodes(nSamples)

    with nogil, parallel():
        for i in prange(nSamples):
            with gil:
                if i % step == 0:
                    state = tuple(model._states)
                    snapshots[state] = snapshots.get(state, 0) + 1 / nSamples
                # model.updateState(next(r))
                model.updateState(r[i])
                pbar.update(1)
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
cpdef dict monteCarlo_alt(object model, dict snapshots,
               long deltas = 10,  long repeats = 11,
               ):
    cdef dict conditional = {}

    cdef double past = time.process_time()


    # # map from node state to idx
    # cdef dict sortr = {i : idx for idx, i in \
    #                    enumerate(model.agentStates)}
    #
    # cdef str nudgeMode = model.__nudgeType
    # cdef double[:] copyNudge = model.__nudges.copy() # store nudges already there
    # # for k in range(repeats):
    # half  = deltas // 2
    # reset = False
    # cdef long[:, :] r = model.sampleNodes(repeats * (deltas + 1))
    # cdef int k, delta, node
    # cdef int counter = 0
    # cdef s = np.array([q for q in snapshots])
    # cdef int N = s.shape[0], n
    # pbar = tqdm(total = N)
    # for n in prange(N, nogil=True, schedule = 'dynamic'):
    #     with gil:
    #         out = np.zeros(\
    #                    (\
    #                     deltas + 1,\
    #                      model._nNodes,\
    #                       model._nStates,\
    #                       ))
    #         startState = s[n]
    #         for k in range(repeats):
    #             # start from the same point
    #             model.states[:] = startState.copy()
    #             model.__nudges[:] = copyNudge.copy()
    #             # print(model.nudges, model.states)
    #             reset        = True
    #
    #             for delta in range(deltas + 1):
    #                 # bin data()
    #                 state = model._states
    #                 # idx   = np.digitize(state, model.agentStates)
    #                 # out[delta, idx] += 1 / repeats
    #                 for node in range(model._nNodes):
    #                     out[delta, node, sortr[state[node]]] += 1 / repeats
    #                     # model.updateState(model.sampleNodes[model.mode](nodeIDs))
    #                 model.updateState(model.sampleNodes(1)[0])
    #                 # model.updateState(r[counter])
    #                 # counter += 1
    #                 # check if pulse turn-off
    #                 if reset:
    #                     if nudgeMode == 'pulse' or nudgeMode == 'constant' and delta >= half:
    #                         model.__nudges[:] = copyNudge.copy()
    #                         reset =  False
    #         conditional[tuple(s[n])] = out
    #         pbar.update(1)
    cdef func = functools.partial(\
                                parallelMonteCarlo_alt, model = model,
                                repeats = repeats, \
                                deltas = deltas,\
                                )
    cdef np.ndarray value
    cdef tuple key
    cdef s = np.array([k for k in snapshots])
    with mp.Pool(processes = mp.cpu_count()) as p:
        # conditional = p.map(func, tqdm(s))
        # conditional = {k : v for l in p.imap(func,tqdm(s), 5) \
        #                for (k, v) in l.items()}
        for result in p.imap( func, tqdm(s), 1):
            for key, value in result.items():
                conditional[key] = value
    print(f"Delta = {time.process_time() - past}")
    return conditional

@cython.boundscheck(False) # compiler directive
@cython.wraparound(False) # compiler directive
cpdef dict parallelMonteCarlo_alt(\
          long[:] startState, object model, int repeats,\
          int deltas,\
          ):
    cdef dict conditional = {}
    # output time x nodes x node states
    cdef double[:, :, :] out = np.zeros(\
                                   (\
                                    deltas + 1,\
                                     model._nNodes,\
                                      model._nStates,\
                                      ))

    # map from node state to idx
    cdef dict sortr = {i : idx for idx, i in \
                       enumerate(model.agentStates)}

    cdef str nudgeMode = model.__nudgeType
    cdef double[:] copyNudge = model.__nudges.copy() # store nudges already there
    # for k in range(repeats):
    half  = deltas // 2
    reset = False
    cdef long[:, ::1] r = model.sampleNodes(repeats * (deltas + 1))
    cdef int k, delta, node
    cdef int counter = 0
    for k in range(repeats):
        # start from the same point
        model.states[:] = startState.copy()
        model.__nudges[:] = copyNudge.copy()
        # print(model.nudges, model.states)
        reset        = True

        for delta in range(deltas + 1):
            # bin data()
            state = model._states
            # idx   = np.digitize(state, model.agentStates)
            # out[delta, idx] += 1 / repeats
            for node in range(model._nNodes):
                out[delta, node, sortr[state[node]]] += 1 / repeats
                # model.updateState(model.sampleNodes[model.mode](nodeIDs))
                # model.updateState(model.sampleNodes(1)[0])
            model.updateState(r[counter])
            counter += 1
            # check if pulse turn-off
            if reset:
                if nudgeMode == 'pulse' or nudgeMode == 'constant' and delta >= half:
                    model.__nudges[:] = copyNudge.copy()
                    reset =  False
    # print('here')
    return {tuple(startState) : np.asarray(out)}

@cython.boundscheck(False) # compiler directive
@cython.wraparound(False) # compiler directive
cpdef mutualInformation_alt(dict conditional, int deltas, \
                          dict snapshots, object model):
    '''
    Returns the node distribution and the mutual information decay
    '''
    px = np.zeros((deltas + 1, model._nNodes, model._nStates))
    H  = np.zeros((deltas + 1, model._nNodes))
    for key, p in conditional.items():
        H  += np.nansum(p * np.log2(p), -1) * snapshots[key]
        px += p * snapshots[key] # update node distribution
    H -= np.nansum(px * np.log2(px), -1)
    return px, H
