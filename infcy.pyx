# cython: infer_types=True
# distutils: language=c++
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

cdef int _CORE = 1 # for imap
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
cpdef getSnapShots(object model, int nSamples, int step = 1, int parallel = mp.cpu_count(), int burninSamples = int(1e3)):
    # start sampling
    cdef dict snapshots = {}
    # func = functools.partial(parallelSnapshots, step = step, \
    # burninSamples = burninSamples, model = model)
    #
    func = functools.partial(parallelSnapshots, step = step, \
    burninSamples = burninSamples)

    Z = nSamples

    tmp = []
    for i in range(nSamples):
      m = copy.deepcopy(model)
      m.reset()
      tmp.append((1, m))

    # tmp = np.ones(nSamples)
    # tmp = [nSamples]
    with mp.Pool(processes = parallel) as p:
        results = p.imap(func, tqdm(tmp), 1)
        for result in results:
          for key, value in result.items():
            snapshots[key] = snapshots.get(key, 0) + value / Z
    print(f'Found {len(snapshots)} states')
    return snapshots

    # def parallelSnapshots(int nSamples, object model,  int step, int burninSamples):
@cython.boundscheck(False) # compiler directive
@cython.wraparound(False) # compiler directive
def parallelSnapshots(tuple sampleModel,  int step, int burninSamples):
  '''
    Get snapshots by simlating for :nSamples:
    :step: gives how many steps are between samples
  '''
  ### use this if you want to start from random init
  nSamples, model = sampleModel
  model.burnin(burninSamples) # start from random state
  cdef dict snap = {}
  # init generator
  cdef int n = int(nSamples * step)
  # pre-define what to sample
  # r = (\
  # model.sampleNodes[model.mode](model.nodeIDs)\
  # for i in range(n)\
  #     )
  # cdef np.ndarray r = np.array([\
  # model.sampleNodes[model.mode](model.nodeIDs)] for _ in range(n))
  cdef object r = (model.sampleNodes[model.mode](model.nodeIDs) for _ in range(n))
  # for i in range(n))
  # simulate
  cdef int i
  for i in range(n):
      if i % step == 0:
          state = model.states
          snap[tuple(state)] = snap.get(tuple(state), 0) + 1
      # model.updateState(model.sampleNodes[model.mode](model.nodeIDs))
      # model.updateState(r[i])
      model.updateState(next(r))
  return snap

@cython.boundscheck(False) # compiler directive
@cython.wraparound(False) # compiler directive
def monteCarlo(object model, dict snapshots,  dict conditions,
               long deltas = 10,  long repeats = 11,
               bint parallel = True, \
               dict pulse = {}, str mode = 'source'):
    '''
    Calls parallel montecarlo. Performs monte carlo samples based on the
    parsed snapshots.

    Input:
        :repeats: Number of times to perform monte-carlo sample per snapshot
        :deltas: number of time steps to simulate for
        :use_more_power: whether or not to use multiple cores
        :pulse: a dictionary indicating which nodes to inject an external pulse.
        A pulse only lasts for the initial time step (delta 0  > 1)
    '''
    # would prefer to make this a generator, however then mp breaks
    print('Starting MC sampling')
    func = functools.partial(\
                            parallelMonteCarlo, model = model,
                            mode = mode, repeats = repeats, \
                            deltas = deltas, snapshots = snapshots, conditions = conditions,\
                            pulse = pulse\
                            )
    conditional = {}
    with mp.Pool(mp.cpu_count()) as p:
        for result in p.imap( func, tqdm(snapshots), 10):
            for key, value in result.items():
                conditional[key] = conditional.get(key, 0) + value
    return conditional
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
cpdef monteCarlo_alt(object model, dict snapshots,
               long deltas = 10,  long repeats = 11,
               bint parallel = True, \
               dict pulse = {}, str mode = 'source'):



   print('Starting MC sampling')
   func = functools.partial(\
                            parallelMonteCarlo_alt, model = model,
                            mode = mode, repeats = repeats, \
                            deltas = deltas, snapshots = snapshots,\
                            pulse = pulse\
                            )
   conditional = {}
   cdef np.ndarray value
   cdef tuple key
   cdef s = np.array([k for k in snapshots])
   print(s.shape)
   with mp.Pool(mp.cpu_count()) as p:
       for result in p.imap( func, tqdm(s), _CORE):
           for key, value in result.items():
               conditional[key] = value
   return conditional

@cython.boundscheck(False) # compiler directive
@cython.wraparound(False) # compiler directive
cpdef dict parallelMonteCarlo_alt(\
          long[:] startState, object model, int repeats,\
          int deltas,\
          dict pulse, dict snapshots, str mode):
  # global model, repeats, conditions, deltas, pulse, snapshots, mode
  # convert to binary state from decimal
  # flip list due to binary encoding
  # if isinstance(startState, tuple):
  # cdef np.ndarray [:] startState = np.array(startState, dtype = model.statesDtype) # convert back to numpy array
  # r = (model.sampleNodes[model.mode](model.nodeIDs) for i in range(repeats * deltas))
  cdef dict conditional = {}


  # output time x nodes x node states
  cdef np.ndarray out = np.zeros((deltas + 1, model.nNodes, model.nStates))

  # map from node state to idx
  cdef dict sortr = {i : idx for idx, i in enumerate(model.agentStates)}

  # declarations

  # cdef object nodesToUpdate = \
  # (\
  # model.sampleNodes[model.mode](model.nodeIDs) for _ in range((deltas + 1) * repeats)\
  # )

  cdef int k, delta, node
  cdef dict _pulse # tmp storage
  cdef int [:] nodeIDs  = model.nodeIDs
  cdef np.ndarray copyNudge
  # cdef np.ndarray r = np.array([\
  # model.sampleNodes[model.mode](model.nodeIDs) for _ in range((deltas + 1)* repeats)  ])
  # cdef object r = (model.sampleNodes[model.mode](model.nodeIDs) for _ in range((delta + 1) * repeats))
  cdef int counter = 0
  cdef str nudgeMode = model.nudgeMode
  for k in range(repeats):
    # start from the same point
    model.states = np.array(startState.copy(), dtype = model.statesDtype)
    _pulse = copy.copy(pulse)
    # print(pulse)

    tmp = model.simulate(nSamples = deltas, step = 1, pulse = pulse) # returns delta + 1 x node
    # for idx, state in enumerate(tmp):
    #   for node in nodeIDs:
    #     nodeState    = state[node]
    #     nodeStateIdx = sortr[nodeState]
    #     out[idx, node, nodeStateIdx] += 1 / repeats

    for delta in range(deltas + 1):
      # bin data
      state = model.states
      for node, state in enumerate(state):
        out[delta, node, sortr[state]] += 1 / float(repeats)
      # assign nudges if present
      if _pulse:
        copyNudge = model.nudges.copy() # story nudges already there
        for node, nudge in _pulse.items():
          model.nudges[model.mapping[node]] = nudge # TODO: either set or add
    #
    #   # update model
      # r = next(nodesToUpdate)
      model.updateState(model.sampleNodes[model.mode](nodeIDs))
      # model.updateState(r[counter])
      # model.updateState(next(r))
      counter += 1
      # check if pulse turn-off
      if _pulse:
        model.nudges = copyNudge
        # check stop conditions
        conditions = (nudgeMode == 'constant' and delta >= deltas // 2,\
                      nudgeMode == 'pulse')
        if nudgeMode == 'pulse':
          _pulse = {}
      elif nudgeMode == 'constant' and delta >= deltas // 2:
        # if any(conditions):
          _pulse = {}

  return {tuple(startState) : out}

@cython.boundscheck(False) # compiler directive
@cython.wraparound(False) # compiler directive
cpdef mutualInformation_alt(dict conditional, int deltas, \
                          dict snapshots, object model):
  '''
   Returns the node distribution and the mutual information decay
  '''

  # conditional is a conditional here
  # loop declaration
  # cdef tuple key
  # cdef int delta
  px = np.zeros((deltas + 1, model.nNodes, model.nStates))
  H = np.zeros((deltas + 1, model.nNodes))
  for key, p in conditional.items():
    H  += np.nansum(p * np.log2(p), -1) * snapshots[key]
    px += p * snapshots[key] # update node distribution
  H -= np.nansum(px * np.log2(px), -1)
  return px, H

@cython.boundscheck(False) # compiler directive
@cython.wraparound(False) # compiler directive
def parallelMonteCarlo(startState, model, repeats, snapshots, conditions, deltas,  pulse, mode):
  # def parallelMonteCarlo(startState):
    '''
    parallized version of computing the conditional, this can be run
    as a single core version
cowan
    :mode: sinc or source. source mode shifts the conditions[0], sinc mode shift the
    the conditions[1]. De facto the first index is the state and the second
    index is the node.

    '''
    doPulse = True if pulse != {} else False# prevent empty of entire pulse
    # convert to binary state from decimal
    # flip list due to binary encoding
    if type(startState) is tuple:
        startState = np.array(startState, dtype = mode.statesDtype)
    r = (model.sampleNodes[model.mode](model.nodeIDs) for i in range(repeats * deltas))
    cdef dict conditional = {}

    cdef int k
    cdef int delta
    cdef int j
    for k in range(repeats):
        # start from the same point
        model.states = startState.copy()
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

                ty = encodeState(Y, model.nStates)
                tx = encodeState(X, model.nStates)
                tmp = (idx, tx, ty, delta)
                #  bin data
                conditional[tmp] = conditional.get(tmp, 0) + \
                snapshots[tuple(startState)] / repeats
            model.updateState(next(r))# update to next step
            # restore pulse
            if doPulse:
                model.nudges = copynudge
                if model.nudgeMode == 'pulse':
                  doPulse = False
    return conditional

@cython.boundscheck(False)
@cython.wraparound(False)
def reverseCalculation(int nSamples, object model, int delta, dict pulse):
  '''
  Computes MI based on slice of time data (indicated by :delta:)
  '''
  res   = model.simulate(nSamples, verbose = True, pulse = pulse)
  n     = res.shape[0]
  cond  = {} # storage for conditional distribution
  cc    = {} # counter for the unique states
  state = {} # gather the unique states

  Z   = n - delta # number of samples
  jdx = 1
  for i in tqdm(range(delta, n + 1 - jdx)):
      tmp      = res[i - delta: i + jdx] # get time slice relative to the current
      x        = tuple(res[i])           # this is the state considered
      state[x] = state.get(x, 0) + 1 / Z # normalize its probability and count
      current  = cond.get(x, np.zeros((*tmp.shape, model.nStates)))
      binned   = np.digitize(tmp, model.agentStates, right = True)
      for i in range(delta + 1):
        for j in range(model.nNodes):
            current[i, j, binned[i,j]] += 1
      cond[x]  = current
      cc[x]    = cc.get(x, 0) + 1 # count the bins
  print(f'unique states {len(cond)}')

  H  = np.zeros((delta + jdx, model.nNodes))
  px = np.zeros((delta + jdx, model.nNodes, model.nStates))
  conditional = {}
  for key, value in tqdm(cond.items()):
      z                = value / cc[key]  # how many times does the state occur?
      conditional[key] = z
      # assert all(z + 1 - z) == 1
      # z[np.isnan(z)] = 0
      px += value / Z
      x  = np.nansum(z * np.log2(z), axis = -1) # sum over node states
      H  += state[key] * x
  tmp = np.nansum(px * np.log2(px), axis = -1)
  H  -= tmp
  return res, cc, px, conditional, state, H

@cython.boundscheck(False) # compiler directive
@cython.wraparound(False) # compiler directive
def mutualInformation(dict conditional, condition, int deltas):
    '''Condition is the idx to the dict'''
    cdef dict data = {\
    key : value for key, value in conditional.items() if key[0] == condition}
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
        H[delta] += map(sum, (py[delta][state] * v * np.log2(v) \
                              for v in value.values() \
                              for state, value in pxy[delta].items if v > 0))
        # for state, value in pxy[delta].items():
        #     for node, v in value.items():
        #         if v > 0:
        #             H[delta] += py[delta][state] * v * np.log2(v)
    return H.T
def encodeState(state, nStates):
    return int(''.join(format(int(i), f'0{nStates - 1}b') for i in state), 2)
def decodeState(state, nStates, nNodes):
    tmp = format(state, f'0{(nStates -1 ) * nNodes}b')
    n   = len(tmp)
    nn  = nStates - 1
    return np.array([int(tmp[i : i + nn], 2) for i in range(0, n, nn )], dtype = INT16)
