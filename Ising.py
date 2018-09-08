__author__ = 'Casper van Elteren'
__version__ = '1'
from numpy import *
from random import sample
import networkx as nx, ties,  copy, itertools
from tqdm import tqdm, tqdm_notebook, tnrange

# TODO: I think states are being aliased
#%load_ext Cython
# TODO: remove plotting to separate module
# TODO: add nudgin
#%%cython
class Ising:
    def __init__(self, g = None, beta = None, start = None):
        '''
        Constructor
        :: g = nx graph type
        :: beta = 1/temperature
        ::  start = starting state ; can be used to reason from a state
        '''
        self.g = g if g != None else nx.complete_graph(64)
        self.beta = beta if beta != None else .4 # temperature

        # init the graph with random initial state
        if type(start) is  type(None):
            states = {}
            for node in self.g.nodes():
                    states[node] = 1 if random.rand() < .5 else -1
        else:
#            print('here')
#            print(type(start))
            assert type(start) is dict
            states = start

        # TODO : make this part of the input
        for (i,j) in self.g.edges():
            self.g[i][j]['J'] = 1
#            self.g[i][j]['overlap'] = ties.computeOverlap(i, j, self.g)
        self.states = states

    def energy(self, node, states = None, state = None):
        '''
        Returns the sum of the neigbor states of : node
        Note this does not return exp(-)

        takes states and state as input for computing of the time shifted probabilities
        '''
        # find neighbors of the nodes
        states    = self.states if type(states) == type(None) else states
#        if type(states) == .
        neighbors = self.g.neighbors(node) # thisremains constant
        state     = states[node] if type(state) == type(None) else state


        assert type(states) is dict

        # compute the energy and decide to update
        energy =  - sum(\
        [states[neighbor] * self.g[node][neighbor]['J'] * state for neighbor in neighbors]\
                      )
        energy_flip = -sum(\
        [states[neighbor] * self.g[node][neighbor]['J'] * -state for neighbor in neighbors]\
                        )
        return energy, energy_flip, state



    def updateState(self, node):
        '''
        Takes as input : node and determines whether to update the state of the node
        Uses MCMC
        '''
        # compute the energies
        energy, flipEnergy, state = self.energy(node)
        # if new energy is less flip
#        p = 1 / (1 + exp(-self.beta * (flipEnergy - energy)))
        p = float_power(1 + exp(-self.beta*(flipEnergy - energy)), -1)
#        if flipEnergy - energy > 0:
#            p = 1 / (1 + exp(-self.beta * (flipEnergy - energy))
#        else:
#            p = 1
        state = -state if random.rand() < p else state

        self.states[node] = state
        tmp = copy.copy(self.states)
        tmp[node] = state
        return tmp

    def step(self):
        '''
        Take a Glauber step
        '''

        # keep sampling if the node is a nudge node; nudge nodes are fixed
        while True:
            node = sample(self.states.keys(), 1)[0]
            if 'nudge' not in self.g.nodes[node]:
                break

        state = self.updateState(node)
        return state

def simulation(model, nDiffStep = int(1e3), nSamples = 500, betaThreshold = 1e-3):
    '''
    Constructs the state probabilities by simulating an Ising model n times
    : nDiffStep : number of steps between sampled states
    : betaThreshold: threshold for the regression analysis for determiniing the burnin samples
    : nSamples: number of states to sample
    returns:
        : nodeProbability: probability of xi being in s == 1 for all xi in X
        : uniqueStates: an array containing the unique state configurations
        : stateProbability: an array containing the state probilities
    '''
         # SETUP ETERS
    magnetization = []                              # used to estimate when sampling from stationary distribution
#    burnin  = int(.1 * n)                          # burnin samples (throw aways)
    uniqueStates = {}                               # keep track of which spaces occur ; only allow these
    nodeProbability = zeros(len(model.g.nodes()))   # stationary distribution for nodes
    print('Construction state probabilities')
    startSample = False                             # first let the system converge
    c = 0
    i = 0

    while c <= nSamples:
#    for i in tqdm(range(nSimSteps)):
        state = model.step()

        # wait to sample from stationary distrbution
        if startSample is not True :
            state = array(list(state.values()))
            magnetization.append(sum(state))
            if i > 50:

                x = arange(len(magnetization))[:, None]
                x = hstack((x, ones(x.shape)))

                # 'running average' (not really)
                y = array(magnetization-cumsum(magnetization)) / (len(x) - 1)
                beta = linalg.pinv(x.T.dot(x)).dot(x.T).dot(y) #NOTE using pseudo-inverse here
#                print(beta, y, x); assert False
                # the linear regression coefficient will be zero if we are in the stationary distribution
                if abs(beta[0]) < betaThreshold : # magic number to allow for drawing line
                    startSample = True
                    print('{0} burnin samples used'.format(i))
                    pbar = tqdm(total = nSamples) # open progressbar
                    # reset counter
                    i = 0
        else:
            if i % nDiffStep == 0:
                state = tuple(state.values()) # reshape to statespace
                nodeProbability += (array(state) + 1) / 2 # only keep probability of being 1
#                tmp = ''.join(str(state))          # hashtable book-keeping
#                print(len(tmp) - 2)
                # count the unique snapshots
                if state not in uniqueStates.keys():
                    uniqueStates[state] = 1
                else:
                    uniqueStates[state] += 1
                c += 1
                pbar.update(1)
        i += 1
    # end for loop
    else:
        pbar.close()
        print('Done with simulation')
        print('Constructing node probabilities')
        nodeProbability /= c
        assert all([0 <= i <= 1 for i in nodeProbability]), 'Probablities are not in the range 0<=x<=1'
        print('Constructing state probabilities')

        Z = sum(list(uniqueStates.values()))                                # normalization constant
        stateProbability = array([i / Z for i in uniqueStates.values()])    # normalize probabilities
#        u = array([fromstring(i[1:-1], dtype = int, sep = ' ') for i in uniqueStates.keys()]) #
        u = array(list(uniqueStates.keys()))
#        print(stateProbability, u)
    return model, u,  stateProbability, nodeProbability

def multualInformationPerDelta(model, beta, nodeProbability, uniqueStates, stateProbability, kSamples, deltas):
    '''
    Returns the multual information for all xi in X per t-di for di in deltas
    : g: graph used in the model
    : beta: inverse temperature of the model
    : uniqueStates: the unique states encoutered after running the simulation
    : kSamples: number of times to simulate per unique state
    : deltas: number of time steps to simulate for
    '''
    # entropy of nodes [node x 1]
    H = -(nodeProbability * log2(nodeProbability) + (1-nodeProbability) * log2(1-nodeProbability))
    print(deltas)
    conditionalDist = []
    print('Estimating MI per node')
    # conditionalDist = number of X^T x deltas x nodes
    for startState in tqdm(uniqueStates):
        # convert to dictionary per node
        startState = {node: i for node, i in zip(model.g.nodes(), startState)}
#        print(startState)
        nodeCond = zeros((deltas, len(nodeProbability)))
#        print('Starting on state')
        for k in range(kSamples):
            model.states = startState.copy() # reste the model states
#            model = Ising(g = g, beta = beta, start = startState.copy()) # TODO: dont
            for delta in range(deltas):
#                if delta == 0:
#                    nodeCond[delta, :] = H
#                else:
                nodeCond[delta, :] += (array(list(model.step().values())) + 1) / 2 # count prob of being 1

        nodeCond /= kSamples
        conditionalDist.append(nodeCond)
    conditionalDist = array(conditionalDist)

    cp = conditionalDist.T # easier writing
    # equation (4) in Appendix 1
    I = (H + nansum(stateProbability * (cp * log2(cp) + (1-cp) * log2(1-cp)), axis = -1).T).T
    return conditionalDist, I

if __name__ == '__main__':
    from matplotlib.pyplot import *

    close('all') # close all figures

    # %% SETUP MODEL
    beta = 1
    nn = 32
#    expectatedDegreeDist = nx.utils.powerlaw_sequence(nn, exponent = 1.5)
#    g = nx.expected_degree_graph(expectatedDegreeDist, selfloops=False)
    g = nx.grid_2d_graph(nn,nn)
#    g = nx.path_graph(nn)
#    g = nx.random_powerlaw_tree(nn, tries = 10000)
    nodes = list(g.nodes())
    [g.remove_node(node) for node in nodes if g.degree(node) == 0 ] #remove singulars


    k = [g.degree(node) for node in g.nodes()]
    uk = unique(k)
    counts = [len(where(k == i)[0]) for i in uk]
    ks     = array(counts)  / sum(counts)
    fig, ax = subplots(); ax.semilogx(uk, ks, '--.', markersize = 10)
    ax.set_xlabel('Degree(k)'); ax.set_ylabel('P(k)')
    print(len(g.nodes()))
    # %%
#    g = nx.path_graph(2) # test case
    model = Ising(g, beta = beta)
    # %% Simulation
    nDiffSteps       = int(10)
    nSamples         = 100
    model, uniqueStates, stateProbability, nodeProbability = simulation(\
                                model, nDiffSteps, nSamples = nSamples, betaThreshold = 1e-2)

    fig, ax = subplots(); ax.imshow(nodeProbability.reshape(nn,nn))
    assert 0
# %% Create conditional probabilities
    kSamples  = 10
    deltas    = 50
    conditionalDist, I = multualInformationPerDelta(model, beta, \
                                                    nodeProbability, uniqueStates, \
                                                    stateProbability, kSamples, deltas)



    # %% absoulte IDT
    epsilon = 1e-4
    idt = []
    for i in I:
        idx = where(abs(diff(i)) < epsilon)[0]
        if len(idx) == 0:
            idx = len(i)
        else:
            idx = idx[0] + 1
        idt.append(idx)
    idt = array(idt)
    k = [model.g.degree(node) for node in model.g.nodes()]
    uk = unique(k)
    idt_per_k = zeros(len(uk))
    for i, ki in enumerate(uk):
        idx = where(k == ki)[0]
        idt_per_k[i] = mean(idt[idx])

    # %%

    fig, ax = subplots();
    ax.plot(uk, int32(idt_per_k),'.')
    config = {'xlabel' : 'degree', 'ylabel': 'idt'}
    setp(ax, **config)

    #%% using scipy to estimate fit


    import scipy
    x = arange(len(I.T))
#    xx = arange()
    def func(x, a, b, c):
        return a + b * exp(-c*x)
    tmp = []
    for i in tqdm(I):
        idx = where(isnan(i))
        i[idx] = 0
        popt, pcov = scipy.optimize.curve_fit(func, x, i, maxfev = int(5 * 1e3)) #returns optimal coefficients
        j = 1
        print (popt)
        yold = func(0, *popt)
        while True:
            y = func(j, *popt)
            if abs(y- yold) < 1e-4:
                tmp.append(j)
                break
            else:
                j +=1
                yold = y
    tmp = array(tmp)


        # %% plotting
    k = list(dict(nx.degree(model.g)).values())
    uk = unique(k)
    # plot state probabilities
    fig, ax = subplots()
    ax.plot(stateProbability)
    ax.set_xlabel('State')
    ax.set_ylabel('Probability')
    savefig('fig1')

    # distribution of IDT
    fig, ax = subplots(); ax.hist(tmp, density = 1);
    config = {'xlabel' : 'idt value', 'ylabel' : 'p(idt)', 'title' : 'Distribution of IDT'}
    setp(ax, **config)
    # %%
    fig, ax = subplots(); ax.plot(I.T)
    setp(ax, 'xlabel', '$t - \delta$', 'ylabel', 'MI')
    print(I)
    k = list(dict(nx.degree(model.g)).values())
    uk = unique(k)
    h = zeros((len(uk), 2))
    for i, ki in enumerate(uk):
        idx = list(map(int, where(k==ki)[0]))
        h[i, :] = [mean(tmp[idx]), std(tmp[idx])]
    #%%\
    counts = [len(where(k == i)[0]) for i in uk]
    ks     = array(counts)  / sum(counts)
    fig, ax = subplots(); ax.plot(uk, ks)
    ax.set_xlabel('Degree(k)'); ax.set_ylabel('P(k)')
    ax.set_title('Degree distribution')
    savefig('Figures/Degree distribution')

    fig, ax = subplots(); ax.errorbar(uk, h[:,0], h[:,-1], )
    ax.set_xlabel('Degree (k)'); ax.set_ylabel('IDT'); ax.set_title('average IDT per degree')
    ax.set_yscale('log');
    savefig('Figures/Average IDT per Degree')

    # %%
    fig, ax = subplots();
    pos = nx.spring_layout(g, weight = 'degree')
    nx.draw_networkx_nodes(g, pos = pos, node_size = 3, ax = ax)
    nx.draw_networkx_edges(g, pos = pos, ax = ax)
    savefig('Figures/Network')

    fig, ax = subplots(); ax.imshow(nx.adj_matrix(g).todense(), aspect = 'auto' )
    savefig('Figures/Adjacency')
    fig, ax = subplots(); j = ax.imshow(I, aspect = 'auto', vmin = 0, vmax = 1)
    colorbar(h, ax=ax)


    # %%
    '''
    For all unique states, for each node do a positive and negative nudge n-times
    '''
    kSamples = 100
    conditionalDist = []
    print('Estimating MI per node')
    # conditionalDist = number of X^T x deltas x nodes
    TAKE_THIS = g.copy(1)
#    assert 0
    storage = zeros((len(TAKE_THIS.nodes()), len(uniqueStates), deltas, len(TAKE_THIS.nodes()) + 1))
    for idx, startState in enumerate(tqdm(uniqueStates)):
        # convert to dictionary per node
        # for each node
        for jdx, node in enumerate(TAKE_THIS.nodes()):
            nudgeGraph = TAKE_THIS.copy(1)
            nudgeGraph.add_node(-1, nudge = 1) # appends node -1 to the list
            nudgeGraph.add_edge(-1, node, J = 1) #TODO: J = weight sorta
#            startState[-1] = 1 # set nudge node to 1 # TODO: also do -1 case
            tmp = hstack((startState, 1))
            tmp = {node: i for node, i in zip(nudgeGraph.nodes(), tmp)}
            model.g = nudgeGraph # update the new graph
            nodeCond = zeros((deltas, len(nodeProbability) + 1))
            for k in range(kSamples):
                model.states = copy.copy(tmp)
                for delta in range(deltas):
                    nodeCond[delta, :] += (array(list(model.step().values())) + 1) / 2 # count prob of being 1

            nodeCond /= kSamples
            storage[jdx, idx, ...] = nodeCond
#        conditionalDist.append(nodeCond)
#    conditionalDist = array(conditionalDist)

    H = storage.reshape(-1, storage.shape[-1]).mean(0)
    H = -array([nansum([i * log2(i), (1-i) * log2(1-i)]) for i in H])
    print(H.shape)
    # equation (4) in Appendix 1
    dd = stateProbability.reshape(1,len(stateProbability), 1, 1)
    cp = storage
    II = nansum(dd * (cp * log2(cp) + (1-cp) * log2(1-cp)), axis = 1).T
    II += H.reshape(len(H), 1, 1)
    # %%
    n = int (1e3)
    testN = zeros(n)
    y = arange(n)
    # %%
    t = time.time()
    for i in range(n):
        idx = random.choice(y)
        testN[idx] += 1


    print(time.time() - t)
    # %%
    t = time.time()
    idx = random.choice(y, n)
    testN[idx] += 1
    print(time.time() - t)
    # %%
    t = time.time()
    list(map(lambda x, idx : x + 1, testN, list(idx)))
    print(t-time.time())

    # %%
    t = time.time()
    [multiply(g,g) for i in range(n)]
    print(time.time() -t )

    t = time.time()
    [g * g for i in range(n)]
    print(time.time() -t )
