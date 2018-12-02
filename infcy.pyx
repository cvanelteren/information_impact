# cython: infer_types=True
# distutils: language=c++
__author__ = 'Casper van Elteren'
import numpy as np
cimport numpy as np
cimport cython

from cython.parallel cimport parallel, prange, threadid
# cimport numpy as np
from libc.stdio cimport printf
# cimport cython
import IO, plotting as plotz, networkx as nx, functools, itertools, platform, pickle,\
fastIsing, copy, time
# from pathos import multiprocessing as mp
import multiprocessing as mp
from tqdm import tqdm   #progress bar
#from joblib import Parallel, delayed, Memory

# TODO: the numpy approach should be re-written in a dictionary only approach in order to prevent memory issues;
# the general outline would be to yield the results and immediately bin them accordingly and write state to disk

# array = functools.partial(np.array, dtype = np.float16) # tmp hack
from libc.stdio cimport printf
cdef int _CORE = 10  # for imap
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

    # pranging
    print(model)
    cdef long[:, :] r = model.sampleNodes(nSamples)
    pbar = tqdm(total = nSamples)

    cdef int i
    with nogil, parallel():
        for i in prange(nSamples):
            with gil:
                if i % step == 0:
                    state = tuple(model.states)
                    snapshots[state] = snapshots.get(state, 0) + 1 / nSamples
                # model.updateState(next(r))
                model.updateState(r[0])
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
@cython.cdivision(True)
cpdef monteCarlo_alt(object model, dict snapshots,
                    long deltas   = 10, \
                    long repeats  = 1000,\
                    ):
    cdef:
        dict conditional = {}
        s = np.array([k for k in snapshots])
        double past = time.process_time()

        np.ndarray value
        tuple key

        dict sortr = {i : idx for idx, i in enumerate(model.agentStates)}
        int N = len(s)
        # long[:, :] r = model.sampleNodes(N * (deltas + 1) * repeats)
        int counter = 0
        str nudgeMode = model.nudgeType
        double[:] copyNudge = model.__nudges.copy() # store nudges already there

        # double[:, :, :] out
    # for k in range(repeats):
    half  = deltas // 2
    reset = False
    cdef dict mc
    cdef int n, node, delta, repeat
    pbar = tqdm(total = N)
    cdef int td = -1
    # with nogil, parallel(num_threads = 4):
    #     td = threadid()
    #     printf('%d ', td)
    #     for n in prange(N):
            # with gil:
    for n in range(N):
        out = np.zeros((deltas + 1, model._nNodes, model._nStates))
        for repeat in range(repeats):

            # start from the same point

            model._states = s[n, :].copy()
            model.__nudges[:] = copyNudge.copy()
            # print(model.nudges, model.states)
            reset        = True

            for delta in range(deltas + 1):
                # bin data()
                state = model.states
                # idx   = np.digitize(state, model.agentStates)
                # out[delta, idx] += 1 / repeats
                for node in range(model._nNodes):
                    out[delta, node, sortr[state[node]]] += 1 /float(repeats)
                    # model.updateState(model.sampleNodes[model.mode](nodeIDs))
                model.updateState(model.sampleNodes(1)[0])
                # model.updateState(r[counter])
                # counter+= 1
                # check if pulse turn-off
                if reset:
                    if nudgeMode == 'pulse' or nudgeMode == 'constant' and delta >= half:
                        model.__nudges[:] = 0
                        reset =  False
        conditional[tuple(s[n])] = out
        pbar.update(1)

    pbar.close()
    print(f"Delta = {time.process_time() - past}")
    return conditional


@cython.boundscheck(False) # compiler directive
@cython.wraparound(False) # compiler directive
cpdef mutualInformation_alt(dict conditional, int deltas, \
                          dict snapshots, object model):
  '''
   Returns the node distribution and the mutual information decay
  '''
  px = np.zeros((deltas + 1, model.nNodes, model.nStates))
  H  = np.zeros((deltas + 1, model.nNodes))
  for key, p in conditional.items():
    H  += np.nansum(p * np.log2(p), -1) * snapshots[key]
    px += p * snapshots[key] # update node distribution
  H -= np.nansum(px * np.log2(px), -1)
  return px, H
