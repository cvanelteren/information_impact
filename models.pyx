# cython: infer_types=True
import numpy as np
cimport numpy as np
import networkx as nx, tqdm, information, functools
import cython
# TODO: maybe add reversemapping [rmap] [done]
# TODO: insert the zero step in the simulate [done]
# TODO: make J weight to make model more general[done]
# TODO: the codes needs to written such that it is model independent; which it currently is\
#however the model does need to have a specific set of constraints [done]
#one of the current constraints is that it needs a base for the agent states
# TODO: add sync, async and single update methods [done]
DTYPE = np.int
ctypedef np.int DTYPE_T
class Model:
    def __init__(self, graph, agentStates, mode = 'async', verbose = False, nudgeMode = 'constant'):
        '''
        General class for the models
        It defines the expected methods for the model; this can be expanded
        to suite your personal needs but the methods defined here need are relied on
        by the rest of the package.

        It translates the networkx graph into numpy dependencies for speed.
        '''
        self.modes = 'sync async single serial' # expand this if we want more modes

        assert mode in self.modes
        # kinda uggly
        statePresent, hPresent, nudgePresent = [\
            True if nx.get_node_attributes(graph, i) != {} else False for i in 'state H nudges'.split(' ')]
        weightPresent = True if nx.get_edge_attributes(graph, 'weight') != {} else False

        agentBins = np.linspace(0, 1, len(agentStates) + 1) # cumsum
        # output checks
        for text, condition in zip('weight state'.split(' '), \
                                   [weightPresent, statePresent]):
            if not condition and verbose:
                print(\
                      'Warning: no {} detected; assuming standard values'.format(\
                                   text)\
                      )
        # model requires weights, nudges, and external field [H]; check for them or assume std values
        for node in graph.nodes():
            if not statePresent:
                r = np.random.rand()
                graph.nodes()[node]['state'] = agentStates[np.digitize(r, agentBins) - 1]
            if not nudgePresent:
                graph.nodes()[node]['nudges'] = 0
        for (i, j) in graph.edges():
            if not weightPresent:
                graph[i][j]['weight'] = 1


        # use graph nodes as nodeID for lattice you get tuples for instance
        mapping = {node: nodeID for nodeID, node in enumerate(graph.nodes())} # hashmap
        if type(graph) is nx.Graph:
            connecting = graph.neighbors
        elif type(graph) is nx.DiGraph:
            connecting = graph.predecessors
        else:
            raise TypeError('Graph type not understood')

        nNodes  = graph.number_of_nodes()
        cdef np.ndarray states     = np.zeros(nNodes)     # store state, H
#        edgeData   = zeros(nNodes, dtype = object)  # contains neighbors and weights [variable length]
        edgeData   = {}
        interaction= {}
        cdef np.ndarray nudges     = np.zeros(nNodes, dtype = object)  # contains the nudge on a node [node x nudges]

        # iterate over the dict, we need to keep track of the mapping
        for node, nodeID in mapping.items():
            states[nodeID] = graph.nodes[node]['state']
            nudges[nodeID] = graph.nodes[node]['nudges']
            neighborData = np.zeros((len(list(connecting(node))), 2)) # store state and J per edge
            for idx, neighbor in enumerate(connecting(node)):
                # directed graphx
                if graph.has_edge(node, neighbor):
                    J = graph[node][neighbor]['weight']
                else:
                    J = graph[neighbor][node]['weight']
                neighborData[idx, :] = mapping[neighbor], J
            edgeData[nodeID]    = np.int32(neighborData[:, 0])
            interaction[nodeID] = neighborData[:, 1]

         # set class data
        cdef np.ndarray nodeIDs  = np.array(list(mapping.values()))
        rmapping = {value : key for key, value in mapping.items()} # TODO: deprecated

        # standard model properties
        # TODO: make some of this private [partly done]
        self.mapping        = mapping
        self.rmapping       = rmapping
        self.states         = states
        self.edgeData       = edgeData     # sparse adjacency [weighted]
        self.interaction    = interaction  # weights
        self.graph          = graph  # sanity storage ; may be need to be removed for very large graphs
        self.nodeIDs        = nodeIDs
        self.nNodes         = nNodes
        self.nudges         = nudges
        self.agentStates    = agentStates
        self.nStates        = len(agentStates)
        self._mode          = mode
        self.nudgeMode      = nudgeMode
        self.sampleNodes    = dict(\
                                single = functools.partial(np.random.choice, size = 1),\
                                async= functools.partial(np.random.choice, size = self.nNodes, replace = False),\
                                sync  = functools.partial(np.random.choice, size = self.nNodes, replace = False),\
                                serial = np.sort\
                                )
    # deprecated as it decreases performance, this functionality should be added
    # to a function that loops through all possible states
#    @property
#    def states(self): return self._states
    # list setters
#    @states.setter
#    def states(self, value):
#        '''
#        Checks if new state is correct size, and numpy array
#        '''
#        if len(value) != self.nNodes:
#            raise ValueError('New state is not correct size {}'.format(self.nNodes))
#        elif type(value) is ndarray:
#            self._states = value
#        else:
#            raise TypeError('type is not numpy array')
    @property
    def mode(self):   return self._mode

    @mode.setter
    def mode(self, value):
        '''
        When adjusting the mode, it checks if the mode is possible
        '''
        if value not in self.modes:
            raise ValueError('Mode is not recognized, please use {}'.format(self.modes))
        else:
            self._mode = value

    # list mandatory functions
    def updateState(self):
        '''This method is required defaulting to zero'''
        assert False, 'model should implement this function'

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    def simulate(self, \
                 int nSamples = 100, int step = 1, pulse = {}, verbose = False):
        '''
        Returns state after sampling
        Input:
            : nSamples: number of required samples
            : step    : steps between samples
            : pulse   : a dict containing node to nudge at t - 1. Note this is the networkx node name not idx
        returns:
            :simulationResults: samples x states; note minimum samples is always 2
        '''
        # pre-define number of simulation steps
#        nodesToUpdate = random.choice(self.nodeIDs, size = (nSamples * step + 1, self.nNodes if self.mode != 'single' else 1))        # this seems faster  ^ when no while loop...
        if verbose:
            print(\
              'Simulation parameters\n number of samples: {}'\
              '\n step size {}\n pulse : {}'.format(\
                                                    nSamples, step, pulse\
                                                    )\
                            )
        # this was first done inside update; pre-definition seems faster
        # haven't explicitly tested the generator variant, but sense will reduce memory loads
        nodesToUpdate = \
        (self.sampleNodes[self.mode](self.nodeIDs) for _ in range(nSamples * step + 1)) # convert to generator
        # nodesToUpdate = np.array([self.sampleNodes[self.mode](self.nodeIDs) for i in range(nSamples * step + 1)])
        # init storage vector
        simulationResults         = np.zeros( (nSamples + 1, self.nNodes), dtype = self.states.dtype) # TODO: this should be a generator as well

        # cdef long [:] state = self.states
        # cdef long [:, ::1] sr = simulationResults
        # cdef int n         = int(self.nNodes)

        # for i in range(n):
          # sr[0, i] = state[i]

        simulationResults[0, :]   = self.states # always store current state
        sampleCounter, stepCounter= 1, 1 # zero step is already done?
        if verbose : pbar = tqdm.tqdm(total = nSamples) # init progressbar
        while sampleCounter <= nSamples:
            if pulse: # run pulse; remove nudges after sim step
                copyNudge = self.nudges.copy() # make deep copy  #TODO: think this will cause slowdowns for very large networks
                for node, nudge in pulse.items():
                    if type(node) is tuple:
                        for i in node:
                            self.nudges[self.mapping[i]] = nudge
                    else:
                        # inject overwhelming pulse for 1 delta
                        if self.nudgeMode == 'pulse':
                          state = self.states[self.mapping[node]]
                          self.nudges[self.mapping[node]] =  abs(nudge) if state == -1 else -abs(nudge)
                        # constant pulse add
                        else:
                          self.nudges[self.mapping[node]] = nudge
            self.updateState(next(nodesToUpdate))
            # self.updateState(nodesToUpdate.__next__()) # update generator
            # twice swap cuz otherway was annoying
            if pulse:
                self.nudges = copyNudge # reset the copy
                if self.nudgeMode == 'pulse':
                  pulse = {}
                elif self.nudgeMode == 'constant' and sampleCounter == nSamples // 2:
                  pulse = {}
            if stepCounter % step == 0: # if stepCounter starts at zero 1 step is already made
                simulationResults[sampleCounter, :] = self.states
                # for i in range(n):
                  # sr[sampleCounter, i] = state[i]
                sampleCounter += 1 # update sample counter
                if verbose: pbar.update(1)  # update PB
            stepCounter += 1  # update step counter
        # out of while
        else:
            return simulationResults
