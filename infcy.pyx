# cython: infer_types=True
__author__ = 'Casper van Elteren'
import numpy as np
cimport numpy as np
import cython, copy
from cython import parallel
# cimport numpy as np
# cimport cython
import IO, plotting as plotz, networkx as nx, functools, itertools, platform, pickle,\
fastIsing
# from pathos import multiprocessing as mp
import multiprocessing as mp
from tqdm import tqdm   #progress bar
#from joblib import Parallel, delayed, Memory

# TODO: the numpy approach should be re-written in a dictionary only approach in order to prevent memory issues;
# the general outline would be to yield the results and immediately bin them accordingly and write state to disk

# array = functools.partial(np.array, dtype = np.float16) # tmp hack

cdef int _CORE = 500 # for imap
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
def getSnapShots(object model, int nSamples, int step = 1, int parallel = mp.cpu_count(), int burninSamples = int(1e3)):
    # start sampling
    cdef dict snapshots = {}
    func = functools.partial(parallelSnapshots, step = step, \
    burninSamples = burninSamples, model = model)

    # func = functools.partial(parallelSnapshots, step = step, \
    # burninSamples = burninSamples)

    Z = nSamples

    # tmp = []
    # for i in range(nSamples):
    #   m = copy.deepcopy(model)
    #   m.reset()
    #   tmp.append((1, m))

    # tmp = np.ones(nSamples)
    tmp = [nSamples]
    with mp.Pool(processes = parallel) as p:
        results = p.imap(func, tqdm(tmp), 1)
        for result in results:
          for key, value in result.items():
            snapshots[key] = snapshots.get(key, 0) + value / Z
    print(f'Found {len(snapshots)} states')
    return snapshots

@cython.boundscheck(False) # compiler directive
@cython.wraparound(False) # compiler directive
def parallelSnapshots(int nSamples, object model,  int step, int burninSamples):
# def parallelSnapshots(tuple sampleModel,  int step, int burninSamples):
    '''
    Get snapshots by simlating for :nSamples:
    :step: gives how many steps are between samples
    '''
    ### use this if you want to start from random init
    # nSamples, model = sampleModel
    # model.burnin(burninSamples) # start from random state


    cdef dict snap = {}
    # init generator
    cdef int n = int(nSamples * step)
    # pre-define what to sample
    r = (\
    model.sampleNodes[model.mode](model.nodeIDs)\
    for i in range(n)\
        )

    # simulate
    cdef int i
    for i in range(n):
        if i % step == 0:
            state = model.states
            # state = state if np.sign(state.mean()) < 0 else -state # local dynamics
            snap[tuple(state)] = snap.get(tuple(state), 0) + 1
        model.updateState(next(r))
    return snap

@cython.boundscheck(False) # compiler directive
@cython.wraparound(False) # compiler directive
def monteCarlo(object model, dict snapshots,  dict conditions,
               long deltas = 10,  long kSamples = 11,
               bint parallel = True, \
               dict pulse = {}, str mode = 'source'):
    '''
    Calls parallel montecarlo. Performs monte carlo samples based on the
    parsed snapshots.

    Input:
        :kSamples: Number of times to perform monte-carlo sample per snapshot
        :deltas: number of time steps to simulate for
        :use_more_power: whether or not to use multiple cores
        :pulse: a dictionary indicating which nodes to inject an external pulse.
        A pulse only lasts for the initial time step (delta 0  > 1)
    '''
    # would prefer to make this a generator, however then mp breaks



    print('Starting MC sampling')
    # uggly temp for testing if partial is a bottle neck
    globals()['kSamples'] = kSamples
    globals()['deltas'] = deltas
    globals()['conditions'] = conditions
    globals()['pulse'] = pulse
    globals()['mode'] = mode
    globals()['model'] = model
    globals()['snapshots'] = snapshots
#    func = functools.partial(\
#    parallelMonteCarlo, model = model,
#    mode = mode, kSamples = kSamples, \
#    deltas = deltas, snapshots = snapshots, conditions = conditions,\
#    pulse = pulse)
    joint = {}
    with mp.Pool(mp.cpu_count()) as p:
        for result in p.imap( parallelMonteCarlo, tqdm(snapshots), 10):
            for key, value in result.items():
                joint[key] = joint.get(key, 0) + value
    return joint
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
def monteCarlo_alt(object model, dict snapshots,
               long deltas = 10,  long kSamples = 11,
               bint parallel = True, \
               dict pulse = {}, str mode = 'source'):



   print('Starting MC sampling')
   # uggly temp for testing if partial is a bottle neck
   globals()['kSamples'] = kSamples
   globals()['deltas'] = deltas
   # globals()['conditions'] = conditions
   globals()['pulse'] = pulse
   globals()['mode'] = mode
   globals()['model'] = model
   globals()['snapshots'] = snapshots
  #    func = functools.partial(\
  #    parallelMonteCarlo, model = model,
  #    mode = mode, kSamples = kSamples, \
  #    deltas = deltas, snapshots = snapshots, conditions = conditions,\
  #    pulse = pulse)
   joint = {}
   with mp.Pool(mp.cpu_count()) as p:
       for result in p.imap( parallelMonteCarlo_alt, tqdm(snapshots), 10):
           for key, value in result.items():
               joint[key] = value
   return joint

@cython.boundscheck(False) # compiler directive
@cython.wraparound(False) # compiler directive
def parallelMonteCarlo_alt(startState):
  global model, kSamples, conditions, deltas, pulse, snapshots, mode
  # convert to binary state from decimal
  # flip list due to binary encoding
  if type(startState) is tuple:
      startState = np.array(startState, dtype = INT16) # convert back to numpy array
  r = (model.sampleNodes[model.mode](model.nodeIDs) for i in range(kSamples * deltas))
  cdef dict conditional = {}

  cdef int k
  cdef int delta
  cdef int j
  cdef long [:] nodeIDs  = model.nodeIDs
  out = np.zeros((deltas + 1, model.nNodes, len(model.agentStates)))
  sortr = {i : idx for idx, i in enumerate(model.agentStates)}

  cdef int noteState
  cdef int node
  cdef int nodeStateIdx
  cdef np.ndarray tmp

  for k in range(kSamples):
      # start from the same point
      model.states = np.array(startState.copy(), dtype = model.states.dtype)
      # res = model.simulate(deltas, 1, pulse)
      tmp = model.simulate(nSamples = deltas, step = 1, pulse = pulse) # returns delta + 1 x node
      for idx, state in enumerate(tmp):
        # state = state if np.sign(state.mean()) < 0 else -state # local dynamics
        for node in nodeIDs:
            nodeState = state[node]
            nodeStateIdx = sortr[nodeState]
            out[idx, node, nodeStateIdx] += 1/kSamples
  return {tuple(startState) : out}

@cython.boundscheck(False) # compiler directive
@cython.wraparound(False) # compiler directive
def mutualInformation_alt(conditional):
  '''Condition is the idx to the dict'''

  cdef np.ndarray H = np.zeros(deltas)
  # joint is a conditional here
  # loop declaration
  # cdef tuple key
  # cdef int delta
  px = np.zeros((deltas + 1, model.nNodes, len(model.agentStates)))
  H = np.zeros((deltas + 1, model.nNodes))
  for key, value in conditional.items():
    H += snapshots[key] * np.nansum(value * np.log2(value), -1)
    px += value * snapshots[key]
  H -= np.nansum(px * np.log2(px), -1)
  return H

@cython.boundscheck(False) # compiler directive
@cython.wraparound(False) # compiler directive
#def parallelMonteCarlo(startState, model, kSamples, snapshots, conditions, deltas,  pulse, mode):
def parallelMonteCarlo(startState):
    '''
    parallized version of computing the conditional, this can be run
    as a single core version
cowan
    :mode: sinc or source. source mode shifts the conditions[0], sinc mode shift the
    the conditions[1]. De facto the first index is the state and the second
    index is the node.

    '''
    global model, kSamples, conditions, deltas, pulse, snapshots, mode
    doPulse = True if pulse != {} else False# prevent empty of entire pulse
    # convert to binary state from decimal
    # flip list due to binary encoding
    if type(startState) is tuple:
        startState = np.array(startState, dtype = INT16) # convert back to numpy array
    r = (model.sampleNodes[model.mode](model.nodeIDs) for i in range(kSamples * deltas))
    cdef dict conditional = {}

    cdef int k
    cdef int delta
    cdef int j
    for k in range(kSamples):
        # start from the same point
        model.states = np.array(startState.copy(), dtype = model.states.dtype)
        # res = model.simulate(deltas, 1, pulse)

        for delta in range(deltas):
            # bin the output
            if doPulse:
                copynudge = model.nudges.copy() # copy if perma nudge present (option
                # overwrite nudge
                for key, value in pulse.items():
                    model.nudges[key] = value

            state          = model.states
#            targetState    = startState if mode == 'sinc' else state
#            conditionState = state      if mode == 'sinc' else startState

            for condition, idx in conditions.items():
                # collect bins
                Y = tuple((startState[j] + 1) / 2 for j in condition[1])
                X = tuple((state[j] + 1) / 2 for j in condition[0])
#                Y = tuple((conditionState[j] + 1) / 2 for j in condition[1])
#                X = tuple((targetState[j] + 1) / 2 for j in condition[0])
                # Y = conditionState[list(condition[1])]
                # X = targetState[list(condition[0])]

                ty = encodeState(Y, 2)
                tx = encodeState(X, 2)
                tmp = (idx, tx, ty, delta)
                #  bin data
                conditional[tmp] = conditional.get(tmp, 0) + \
                snapshots[tuple(startState)] / kSamples
            model.updateState(next(r))# update to next step
            # restore pulse
            if doPulse:
                model.nudges = copynudge
                if model.nudgeMode == 'pulse':
                  doPulse = False
    return conditional
@cython.boundscheck(False) # compiler directive
@cython.wraparound(False) # compiler directive
def mutualInformation(dict joint, int condition, int deltas):
    '''Condition is the idx to the dict'''
    cdef dict data = {\
    key : value for key, value in joint.items() if key[0] == condition}
    cdef dict px = {i : {} for i in range(deltas)}
    cdef dict py = {i : {} for i in range(deltas)}
    cdef np.ndarray H = np.zeros(deltas)

    # loop declaration
    # cdef tuple key
    # cdef int delta
    for key, value in tqdm(data.items()):

      delta = key[3]
        # bin regardless of key 1
      newkey = key[2]
      px[delta][newkey] = px[delta].get(newkey, 0) + value

        # py
      newkey = key[1]
      py[delta][newkey] = py[delta].get(newkey, 0) + value
    # conditional
    cdef dict pxy = {i : {j : {} for j in py[i].keys()} for i in range(deltas)}
    for key, value in tqdm(data.items()):
        state, node, delta = key[1], key[2], key[3]
        pxy[delta][state][node] = pxy[delta][state].get(node, 0) + value / py[delta][state]

    # loop declaration
    # loop
    for delta in tqdm(range(deltas)):
        H[delta] -= np.nansum([p * np.log2(p) for p in px[delta].values()])
        for state, value in pxy[delta].items():
            for node, v in value.items():
                if v > 0:
                    H[delta] += py[delta][state] * v * np.log2(v)
    return H
def encodeState(state, nStates):
    return int(''.join(format(int(i), f'0{nStates - 1}b') for i in state), 2)
def decodeState(state, nStates, nNodes):
    tmp = format(state, f'0{(nStates -1 ) * nNodes}b')
    n   = len(tmp)
    nn  = nStates - 1
    return np.array([int(tmp[i : i + nn], 2) for i in range(0, n, nn )], dtype = INT16)
