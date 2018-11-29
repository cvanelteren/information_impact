import h5py, numpy as np, multiprocessing as mp, functools, itertools
from numba import jit
from tqdm import tqdm
import tempfile
def compute(func, fileName, sourceData, targetData, nCores = mp.cpu_count()):
    '''
    General computation using files
    '''
    # had to move things around in a very uggly way due to how i organized
    # the data [future me problems]
    if nCores > 1: print('Spawning parallel processes')

    with h5py.File(fileName, updateType= 'a') as f: # creates file if it doesn't exist
        if targetData in f:
            print('Found data; skipping computation')
        else:
            x        = f[sourceData] if type(sourceData) is str else sourceData
            # very uggly... but necessary evil
            # if targetData == 'mc':
            #     temp = tempfile.mktemp()
            #     y = memmap(temp[1], shape = x.shape, dtype = x.dtype)
            #     for i in range(len(x)):
            #         y[i] = x[i]
            #     x = y.reshape(-1, *x.shape[2:]).swapaxes(0, 1)
            # N        = len(x)
            # step     = int(1 / nCores * N)
            # interval = np.arange(0, N, step) # create interval for generator
            # if max(interval) != N - 1:
            #     interval = np.hstack((interval, N - 1))
            print('Starting computation')
            idx = 0 if targetData != 'mc' else 1
            with mp.Pool(nCores) as p:
                # loop through the source compute the target, write to disk
                # warning: chunksize can fill memory, don't set it too low or too high
                for immediateResult in p.imap(func, tqdm(x), 1):
                    if type(immediateResult) is not None:
                        immediateResult = immediateResult[None, ...]
                        if targetData == 'mc':
                            s = immediateResult.shape
                            immediateResult = immediateResult.reshape(-1, *s[2:]).swapaxes(0, 1)
                            # immediateResult = immediateResult.swapaxes(0, 2)
                        if targetData not in f:
                            data = f.create_dataset(targetData, data = immediateResult, \
                                                    dtype = x.dtype,\
                                                    maxshape = tuple(None for _ in immediateResult.shape))
                        else:
                            data.resize(data.shape[idx] + immediateResult.shape[idx], axis = idx) # allow for room
                            if  targetData == 'mc':
                                data[:, -immediateResult.shape[1]:] = immediateResult
#                                print(data.shape, immediateResult.shape)
                            else:
                                data[-immediateResult.shape[0]:] = immediateResult  # append data to stack
    print('Done.')

def chunker(ar, interval):
    for start, stop in zip(interval[:-1], interval[1:]):
        yield ar[start:stop]

def getSnapshots(model, fileName, nSamples, step = 1, parallel = True):

    print('Starting chain sampling')
    func  = functools.partial(getSnapShotsParallel, model = model, \
                              step = step)
    nSamplesCores = np.ones(nSamples, dtype = int) #if parallel else array([nSamples])
    compute(func, fileName, nSamplesCores, 'state') # stores samples in state
    # samples are double if using more cores due to a  `feature`
    tmp = {}
    # we still need to compute the 'unique' state and store them
    # this is storage overhead unfortunately
    with h5py.File(fileName, 'a') as f:
        if 'snapshots' in f:
            print('Found data; skipping computation')
        else:
            for sample in f['state']:
                for s in sample:
                    s = tuple(s)
                    tmp[s] = tmp.get(s, 0) + 1
            z = sum(i for i in tmp.values())

            snapshots   = dict(map(lambda kv: (kv[0], kv[1]/z) , tmp.items()) )
            tmp = []
            for idx, (key, value) in enumerate(snapshots.items()):
                tmp.append(key + (value,))
                # stor = f'snapshots/{np.array(key).tobytes()}'
            f.create_dataset(f'snapshots/',   data = np.array(tmp))
        print(f'Found {f["snapshots"].shape} states')
    # deepdish.io.save(fileName, dict(snapshots = snapshots))
@jit
def getSnapShotsParallel(nSamples, model, step = 100):
    '''
    Toying with the idea of the initial chain obtaining parallel information
    '''
#    print(model.states)
    return model.simulate(nSamples = nSamples, step = step)

def montecarlo(fileName,\
            model, repeats, deltas, pulse = {}, \
            sourceData = 'snapshots/', targetData = 'mc'):
    func = functools.partial(parallelMonteCarlo, model = model, \
    repeats = repeats, deltas = deltas, pulse = pulse)
    compute(func, fileName, sourceData, targetData)

@jit
def parallelMonteCarlo(startState, model, repeats, deltas, pulse = {}):
    '''
    parallized version of computing the conditional, this can be run
    as a single core version
    '''
    # convert to binary state from decimal
    # flip list due to binary encoding
    # print(' in mc', startState)
    value      = startState[-1] # tmp hack; this is the prob value
    startState = startState[:model.nNodes] #
    montecarlo = np.zeros((repeats, deltas + 1, model.nNodes), dtype = int)
    sim = model.simulate # read somewhere that referencing in for loop is slower
    # than pre-definition
    for k in range(repeats):
        model.states       = startState.copy() # prevent alias
        montecarlo[k, ...] = sim(nSamples = deltas, step = 1, pulse = pulse)
    return montecarlo

def mutualInformationShift(model, fileName, repeats, updateType= 'source', \
        sourceData = 'mc', targetData = 'mi', conditions = None, \
        ):
       state2idx = {}
       counter = 0
       with h5py.File(fileName, 'r') as f:
           snapshots = f['snapshots'].value
           # construct mapping of where to bin the states
           for i in f[sourceData].value.reshape(-1, model.nNodes):
               i = tuple(i)
               if i not in state2idx:
                   state2idx[i] = counter
                   counter += 1
           targets = f[sourceData][0, ...]
       # TODO: make this better; this is very confusing.

       if conditions == None:
           conditions = tuple((int(i),) for i in model.nodeIDs)
       condition2idx  = {c : idx for idx, c in enumerate(conditions)}

       N = len(conditions[0])
       # creates possible condition states
       Ystates = tuple(itertools.product(*(model.agentStates for _ in range(N))))
       nodestate2idx       = {i:idx for idx, i in enumerate(Ystates)}

       # init normalizing agent
       Z = {tuple(i[:model.nNodes]) : i[-1] / repeats  for i in snapshots}

       '''for each monte carlo sample we want to bin it into the known samples'''
       # convert state to number; convert conditioned to number


       func = functools.partial(binParallel, \
                                updateType= mode,\
                                mappingCondition = condition2idx,\
                                mappingNode = nodestate2idx,\
                                mappingState = state2idx, \
                                Z = Z,\
                                targets = targets)

       import time
       s = time.time()
       compute(func, fileName, sourceData, 'joint')
       print(time.time() - s)
       # joint = array([\
       # binParallel(\
       # i, mode, mappingCondition, mappingNode, mappingState, Z, targets) for i in montecarloResults])
       with h5py.File(fileName, 'a') as f:
           joint           = f['joint'].value
           pState          = joint.sum(-1) # integrate over nodes
           pNode           = joint.sum(-2) # note this is also x delta when shifting state, but this is just a copy of zero
           pNodeGivenState = joint.copy()  # normalize the conditional from the joint
           pNodeGivenState = joint / pState[..., None]
           I = mutualInformation(pState, pNode, pNodeGivenState)
           f.create_dataset('mi', data = I)
@jit
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
    joint = np.zeros(s) # init joint for speed
    # bin the data
#    print(joint.shape)
    for sampleAtT0, sample in zip(targets, samples):
        X   = sampleAtT0 if updateType== 'sinc' else sample   # t - delta
        Y   = sample if updateType== 'sinc' else sampleAtT0 # t = 0
        jdx = mappingState[tuple(X)] # get correct index for statte
        for condition, idx in mappingCondition.items():
            # zdx =  # slice the correct condition
            zdx =  mappingNode[tuple(Y[list(condition)])]
            joint[idx, jdx, zdx] += Z[tuple(sampleAtT0)]  # get the correct count value
    return joint

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
    p       = np.array(p) if type(p) is not np.ndarray else p # sanity check
    # assert len(p.shape) >= 2, 'p should be atleast 1 x number of states'
    return -np.nansum(p * np.log2(p), axis = -1)
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
    Hc = np.multiply(py, entropy(pxgiveny)).sum(-1) # dunno how to dot in 3D
    return H - Hc
