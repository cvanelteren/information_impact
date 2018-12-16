__author__ = 'Casper van Elteren'
from numpy import *
import IO, plotting as plotz, networkx as nx, functools, itertools, platform, pickle,\
fastIsing, multiprocess as mp, psutil, tempfile
from tqdm import tqdm  #progress bar
from numba import jit
#from joblib import Parallel, delayed, Memory

# TODO: the numpy approach should be re-written in a dictionary only approach in order to prevent memory issues;
# the general outline would be to yield the results and immediately bin them accordingly and write state to disk

array = functools.partial(array, dtype = float16) # tmp hack

# TODO; cut out the samples
#TODO: not finished
def checkArrayFit(x, select = 'gb'):
    base = dict(kb = 1, mb = 2, gb = 3, tb = 4)

    # s = x.size / (bytes * 1024**base[select])
    if type(x) is ndarray:
        bytes = int(''.join(i for i in x.dtype.name if i.isdigit()))
        s = x.size / (bytes * 1024**base[select])
    elif type(x) is tuple:
        i = 1
        for j in x:
            i *= j
        s = i / (32 * 1024**base[select])
    print(s)

def checkDistribution():
    '''Warning statement'''
    from platform import platform
    if 'windows' in platform().lower():
        print('Warning: Windows detected. Please remember to respect the GIL'\
              ' when using multi-core functions')
checkDistribution() # print it only once


def compute(func, x, parallel = True):
    '''
    General compute function with additional checks when running parallel
    Input:
        :parallel: can be a number to indicate the number of cores to use
    '''
    if parallel:
        from contextlib import closing
        print('Spawning parallel processes')
        nCores = mp.cpu_count()
        N = nCores - 1 if type(parallel) is bool else parallel
        if N > nCores:
            N = nCores
        nn = int(1/N * len(x))
        # if the len of iterator is smaller than numbero f cores just do
        # 1 job per core
        if nn < N:
            nn = None

        with mp.Pool(mp.cpu_count()) as p:
            return array(p.map(func, tqdm(x), 100))
        # return array(Parallel(n_jobs=parallel, max_nbytes=None)(delayed(func)(i) for i in tqdm(y)))
                # with closing(mp.Pool(N)) as p:
        #     return array(p.map(func, tqdm(y), 10))
    else:
        return array([func(xi) for xi in tqdm(x)])


def simpleChunk(x, interval):
    '''
    Returns generator of chunked data
    '''
    for start, stop in zip(interval[:-1], interval[1:]):
        yield x[start:stop]

def getSnapShots(model, nSamples, step = 1,  parallel = True):
    '''
    Runs innitial chain for the model to obtain a subset of the
    state space for the model. This cuts down on simulation times but crucially
    ignores states that will occur often.
    Input:
        :model: model, see models
        :nSamples: number of samples to take
        :step: number of steps between samples [to ensure independence]
        :useAllCores: do it in parallel # TODO: UNTESTED
    Returns:
        :snapshots: dict with keys state and value probability
        :node probability: matrix containing the node probabilities per agent state
        :model: the input model [contains the mutated states]
    '''
    #TODO: check this multiprocessing mutation with star map; still need to figure out how to do it quicker
    # the goal is to mutate the results which can be achieved by
    # returning the results from the workers intermittently
    # distribute the samples over the cores
#    useAllCores = False
    print('Starting chain sampling')
    func            = functools.partial(getSnapShotsParallel, model = model, \
                              step = step)
    # hacky trick; in parallel we only perform one step in parallel
    # Note one nees to list if using serial

    nSamplesCores = ones(nSamples, dtype = int) if parallel else [nSamples]
    totalSamples = compute(func, nSamplesCores, parallel)
    totalSamples = totalSamples.reshape(-1, totalSamples.shape[-1]) # in parallel it will return 2 steps by default; this is work-around for this bug

    # normalization
    nodeProb     = zeros( (model.nNodes, len(model.agentStates)) ) # use this for targetting
    mapping      = {i : idx for idx, i in enumerate(model.agentStates)}
    snapshots  = {}
    for sample in totalSamples:
        for node, nodeState in enumerate(sample):
            nodeProb[node, mapping[nodeState]] += 1 # returns + 1
        s = tuple(sample)
        snapshots[s] = snapshots.get(s, 0) + 1
    Z           = sum(i for i in snapshots.values())
    snapshots   = dict(map(lambda kv: (kv[0], kv[1]/Z) , snapshots.items()) )

    Z           = nodeProb.sum(axis = 1)
    nodeProb    = array([i / Z for i in nodeProb.T]).T
    print(f'Done. Found {len(snapshots)} states')
    return snapshots, nodeProb, model

# @jit
def getSnapShotsParallel(nSamples, model, step = 100):
    '''
    Toying with the idea of the initial chain obtaining parallel information
    '''
#    print(model.states)
    return model.simulate(nSamples = nSamples, step = step)


def monteCarlo(model, snapshots,
               deltas = 10, repeats = 11,
               parallel = True, \
               pulse = {}):
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
    print('Starting MC sampling')
    func        = functools.partial(parallelMonteCarlo, model = model, \
                                    repeats = repeats, deltas = deltas, \
                                    pulse = pulse)
    # print(f'Done. Shape={montecarlo}')
    montecarlo = compute(func,  array([i for i in snapshots]), parallel)
    return montecarlo

# @jit
def parallelMonteCarlo(startState, model, repeats, deltas, pulse = {}):
    '''
    parallized version of computing the conditional, this can be run
    as a single core version
    '''
    # convert to binary state from decimal
    # flip list due to binary encoding
    if type(startState) is tuple:
        startState = array(startState) # convert back to numpy array
    montecarlo = zeros((repeats, deltas + 1, model.nNodes))
    sim = model.simulate # read somewhere that referencing in for loop is slower
    # than pre-definition
    for k in range(repeats):
        model.states       = array(startState.copy(), dtype = int) # prevent alias
        montecarlo[k, ...] = sim(nSamples = deltas, step = 1, pulse = pulse)
    return montecarlo
# %%
def mutualInformation(py, px, pxgiveny):
    '''
    Computes :I = H(X) - H(X|Y) = \sum_y p(y) \sum_x p(x | y)  log p(x|y) - \sum_x p(x) log p(x)
    Input:
        :px: probability of x
        :py: probability of y
        :pxgiveny: conditional probability of x given y
    Returns the mutual information between x and y
    '''
    H  = entropy(px)
    Hc = multiply(py, entropy(pxgiveny)).sum(-1) # dunno how to dot in 3D
    return H - Hc


def mutualInformationShift(model, snapshots, montecarloResults,\
                           conditions = None, mode = 'source', parallel = False):
       ''' Computes I(x_i^t ; X^t - delta)
       Input:
           :model: fast ising model
           :nSamples: number of snapshots to get from single run
           :repeats: k monte carlo samples
           :deltas: number of simulation steps to do for the monte carlo samples
       Returns:
           :MI shift: I(x_i^t ; X^t - delta) -> delta x condition x states space x node states
           :pState:  state space probability
           :pNode:  node probability
           :pNodeGivenState: conditional probability
       '''
       # ugly
       # TODO: have to figure out how to handle conditions properly without doing dicts
#             I think the best way would be to run a dict separately, i.e. do multiple runs
#             on the same dataset [done]
       # TODO: shift node and shift state is correct but the variable [semi-done]
       # names are inverted
       # starting to make changes here
       print('Binning data')
       print(montecarloResults.shape)
       nStates, repeats, deltas, nNodes = montecarloResults.shape # TODO: fix this for refactor?
       # look into swapping the axes back to regain performance
       montecarloResults = montecarloResults.reshape(-1, deltas, model.nNodes).swapaxes(0, 1)
       mappingState = {}
       counter = 0
       for sample in montecarloResults.reshape(-1, model.nNodes):
           sample = tuple(sample)
           if sample not in mappingState:
               mappingState[sample] = counter
               counter += 1

       # mappingState      = {tuple(i): idx for idx, i in enumerate(uStates)}
       # TODO: make this better; this is very confusing.
       if conditions == None:
           conditions = tuple((int(i),) for i in model.nodeIDs)
       mappingCondition  = {c : idx for idx, c in enumerate(conditions)}
       N = len(conditions[0])
       # creates possible condition states
       Ystates = tuple(itertools.product(*(model.agentStates for _ in range(N))))
       mappingNode       = {i:idx for idx, i in enumerate(Ystates)}
       # init normalizing agent
       Z = {key : value / (repeats) for key, value in snapshots.items()}
       '''for each monte carlo sample we want to bin it into the known samples'''
       # convert state to number; convert conditioned to number
       targets = montecarloResults[0, ...] # track the zero


       func = functools.partial(binParallel, \
                                mode = mode,\
                                mappingCondition = mappingCondition,\
                                mappingNode = mappingNode,\
                                mappingState = mappingState, \
                                Z = Z,\
                                targets = targets)

       joint = compute(func, montecarloResults, parallel)
       # joint = array([\
       # binParallel(\
       # i, mode, mappingCondition, mappingNode, mappingState, Z, targets) for i in montecarloResults])
       pState          = joint.sum(-1) # integrate over nodes
       pNode           = joint.sum(-2) # note this is also x delta when shifting state, but this is just a copy of zero
       pNodeGivenState = joint.copy()  # normalize the conditional from the joint
       pNodeGivenState = joint / pState[..., None]
       # for i in range(pNodeGivenState.shape[-1]):
           # pNodeGivenState[..., i] /= pState
       I = mutualInformation(pState, pNode, pNodeGivenState)
       return joint, I
# @jit
# @profile
def binParallel(samples, mode, mappingCondition, mappingNode, \
                              mappingState, Z, targets):
    '''
    bins data; aimed to be used in parallel.
    returns joint
    '''
    # TODO: can we hijack this part to make it pairiwse?
    # I would need to giv it different samplesa nd targets
    # shape of joint
    s =  (len(mappingCondition), len(mappingState), len(mappingNode))
    joint = zeros(s) # init joint for speed
    # bin the data

    for sampleAtT0, sample in zip(targets, samples):
        X   = sampleAtT0 if mode == 'sinc' else sample   # t - delta
        Y   = sample if mode == 'sinc' else sampleAtT0 # t = 0
        jdx = mappingState[tuple(X)] # get correct index for statte
        for condition, idx in mappingCondition.items():
            # zdx =  # slice the correct condition
            zdx =  mappingNode[tuple(Y[list(condition)])]
            joint[idx, jdx, zdx] += Z[tuple(sampleAtT0)]  # get the correct count value
    return joint
# %%
def entropy(p):
    '''
    N.B. use this function only for a binary distributed node
    Calculates the entropy of probability distribution
    Input:
        :p: array-like containing the probability distribution, note it assumes that the last axis
        contains the states to be summed over
    return
        :H: entropy of the probability distribution
    '''
    p       = array(p) if type(p) is not array else p # sanity check
    # assert len(p.shape) >= 2, 'p should be atleast 1 x number of states'
    return -nansum(p * log2(p), axis = -1)

def nudgeOnNode(nudgeInfo, model,
                nSamples    = 1000,
                step        = 100,
                deltas      = 10,
                repeats    = 1000,
                mode        = 'source',
                pulse       = False,
                reset       = False,
                parallel    = True):
    '''
    Simulates Ising model by adding nudge on the node
    Input:
        :nudgeInfo: dict with keys node and items the nudge directions, any float +-
        Note: the keys are the names of the node in the graph(!)
        :model: ising model to be used
        :reset: whether to start from random state, will automatically burnin
        :deltas: number of time steps to simulate
        :repeats: number of samples for each time step
    Returns:
        :results: a dict containing node probability, state probability and conditional probability
        per nudge condition (keys), and mutual information.
    '''
    #REMINDER the nudging is now done from a node NAME point of view not the index
    # adding the nudges to the correct places
    # this equivalent to running reset on the model, it looks for the nudginfo
    # if it doesnt find it, it will return 0
    print('Starting nudges\nParameters:')
    # TODO remove this; think about using kwargs in the future
    # print parameters:
    d = dict(nSamples = nSamples, repeats = repeats, step = step,\
             deltas = deltas, mode = mode, reset = reset, pulse = pulse)

    for key, value in d.items():
        print(f'{key:>12}\t\t={value:>12}')
    # add nudge/pulse
    if nudgeInfo and not pulse: # if nudge info is not empty
        for node in model.graph.nodes():
            idx = model.mapping[node] #find the mapping
            model.nudges[idx] = nudgeInfo.get(node, 0)
            if nudgeInfo.get(node, 0) != 0 :
                print(f'Nudging {node} with {nudgeInfo[node]}') # show all the nudges
    # do burnin?
    if reset:
        model.reset(doBurnin = True) # restart from randomstate

    # create state space and node probability
    print('--'*16 + 'Starting simulations' + 16 * '--')
    # construct conditional
    # get snapshots
    snapshots = getSnapShots(model = model, nSamples = nSamples, step = step, parallel = parallel)[0]
    # get mc results
    mr = monteCarlo(model = model, snapshots = snapshots,\
                    deltas = deltas, repeats = repeats,\
                    pulse = nudgeInfo if pulse else {},\
                    parallel = parallel)

    # compute MI
    joint, I = mutualInformationShift(model = model,\
                                                        snapshots = snapshots,\
                                                        montecarloResults = mr,\
                                                        mode = mode,\
                                                        parallel = parallel)
    #TODO: this is a very temporary fix
    try:
        effect = nudgeInfo.get(list(nudge.keys())[0], None)
    except:
        effect = None
    std = dict(repeats = repeats, model = model, nSamples = nSamples,
               deltas = deltas, step = step, pulse = pulse)

    # dict output for organization #TODO: keep this or nah?
    results = {}
    # results['state']         = pState
    # results['node']          = pNode
    # results['conditional']   = pXgivenY/
    results['joint']         = joint
    results['I']             = I
    results['mc']            = mr
    results['snapshots']     = snapshots
    results['model']         = model

    print('--'*16 + 'End of simulations' + 16 * '--')
    return results

#%%
# DEPRECATED
def decToBin(x, base = 2):
    return array([base ** idx * i for idx, i in enumerate(x)]).sum()
def binToDec(x, n, base = 2):
    y = zeros(n)
    t = array([base ** i for i in range(n)])

decToBin((2,2), 3)
# %%

# UNFINISHED
# def transferEntropy(nSamples = 1000,
#                     step    = 100,
#                     repeats = 1000,
#                     L        = 10 ):
#
#     """
#     General outline:
#             The idea is to now not batch a single state with the node,
#             but a range of states now become the bin. Hence we condition on
#             the evolution of the network for some set t - L.
#             We need to:
#                 - Obtain nShapshots simulate for L steps and create a new 'state probability' [batch]
#                 - Keep track of the node probability in the process
#                 - Then sample conditionally from those distributions to get the conditional
#     """
#     # first create the snapshots
#     model, stateSpace, nodeProbability, states = createStateSpace(model, nSamples, step = step)
#     # then we need to extract the node probabilities conditionally for L time
#     conditional     = conditionalProbabilityPerDelta(model           = model,
#                                                      startStates     = stateSpace,
#                                                      L               = deltas,
#                                                      repeats        = repeats)
