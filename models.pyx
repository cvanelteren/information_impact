# cython: infer_types=True
import numpy as np
cimport numpy as np
import networkx as nx, tqdm, information, functools
import cython
from cython import parallel
# TODO: maybe add reversemapping [rmap] [done]
# TODO: insert the zero step in the simulate [done]
# TODO: make J weight to make model more general[done]
# TODO: the codes needs to written such that it is model independent; which it currently is\
#however the model does need to have a specific set of constraints [done]
#one of the current constraints is that it needs a base for the agent states
# TODO: add sync, async and single update methods [done]
DTYPE = np.int
ctypedef np.int DTYPE_T
from libcpp.map cimport map as cmap
from libcpp.string cimport string
from libcpp.vector cimport vector

from cython.operator cimport dereference, preincrement
from libc.stdlib cimport rand
cdef extern from "limits.h":
    double INT_MAX
ctypedef np.ndarray (*UPDATE)(long[:] state, long[:] nodesToUpdate)
cdef class Model:
    # cdef dict __dict__
    cdef public:
        object graph
    def __init__(self, graph, agentStates, mode = 'async', verbose = False, nudgeMode = 'constant'):
        '''
        General class for the models
        It defines the expected methods for the model; this can be expanded
        to suite your personal needs but the methods defined here need are relied on
        by the rest of the package.

        It translates the networkx graph into numpy dependencies for speed.
        '''

        self.modes = 'sync async single serial' # expand this if we want more modes
        # TODO: put this in a constructor
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

        labels = set([type(node)  for node in graph.nodes()])
        assert len(labels) == 1
        # cdef if isinstance(labels[0], str):
        #     cdef string T
        # elif isinstance(labels[0], float):
        #     cdef double T
        # elif isinstance(labels[0], int):
        #     cdef int T
        # use graph nodes as nodeID for lattice you get tuples for instance
        # set class data
        # setup hashmap for internal reference
        cdef dict mapping    = {node  : nodeID for nodeID, node in enumerate(graph.nodes())}
        cdef dict rmapping   = {value : key for key, value in enumerate(graph.nodes())}
        cdef nodeIDs = np.asarray(\
                              list(rmapping.values()), dtype = int)
        # get correct neighbor function
        if type(graph) is nx.Graph:
            connecting = graph.neighbors
        elif type(graph) is nx.DiGraph:
            connecting = graph.predecessors
        else:
            raise TypeError('Graph type not understood')

        cdef int nNodes        = graph.number_of_nodes()
        cdef np.ndarray states = np.zeros(nNodes) # store state, H
#        edgeData   = zeros(nNodes, dtype = object)  # contains neighbors and weights [variable length]
        # cdef vector[int, int] _edgeData
        # cdef vector[int, double] _interaction
        cdef dict _edgeData    = {}
        cdef dict _interaction = {}
        cdef np.ndarray nudges = np.zeros(nNodes, dtype = object) # contains the nudge on a node [node x nudges]

        # iterate over the dict, we need to keep track of the mapping
        for node, nodeID in sorted(mapping.items(), key = lambda x : x[1]):
            states[nodeID] = graph.nodes[node]['state']
            nudges[nodeID] = graph.nodes[node]['nudges']
            neighborData = np.zeros((len(list(connecting(node))), 2)) # store state and J per edge
            for idx, neighbor in enumerate(connecting(node)):
                # directed graphx
                if graph.has_edge(node, neighbor):
                    J = graph[node][neighbor]['weight']
                else:
                    J = graph[neighbor][node]['weight']
                _edgeData[nodeID] = _edgeData.get(nodeID, []) + [mapping[neighbor]]
                _interaction[nodeID] = _interaction.get(nodeID, []) + [J]
                # neighborData[idx, :] = mapping[neighbor], J
            # _edgeData[nodeID]    = neighborData[:, 0]
            # _interaction[nodeID] = neighborData[:, 1]
        # cdef cmap[long, vector[long]] edgeData       = _edgeData
        # cdef cmap[long, vector[double]] interaction  = _interaction
        cdef dict edgeData       = _edgeData
        cdef dict interaction    = _interaction



        # standard model properties
        # TODO: make some of this private [partly done]
        self.mapping        = mapping
        self.rmapping       = rmapping
        self.states         = np.int64(states)
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
        # self.sampleNodes    = dict(\
        #                         single = functools.partial(np.random.choice, size = 1, ),\
        #                         async  = functools.partial(np.random.choice, size = self.nNodes, replace = False),\
        #                         sync   = functools.partial(np.random.choice, size = self.nNodes, replace = False),\
        #                         serial = np.sort\
        #                         )
    # @cython.wraparound(False)
    # @cython.boundscheck(False)
    # def sampleNodes(self, int nSamples, bint fast = True):
    #     cdef np.ndarray samples # return vector
    #     cdef int length # length of target array
    #     cdef np.ndarray sample # tmp storage
    #     cdef np.ndarray nodeIDs = self.nodeIDs
    #     cdef int sampleSize
    #     cdef mode = self.mode
    #     length  = len(nodeIDs)
    #     if mode == 'single':
    #         sampleSize = 1
    #     else:
    #         sampleSize = length
    #     samples = np.ndarray((nSamples, sampleSize), int)
    #     for samplei in range(nSamples):
    #         sample = c_sample(nodeIDs, length, samplei, \
    #                           sampleSize, fast)
    #         samples[samplei, :] = sample
    #     return samples
    # note properties slow down the code a lot
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
    def updateState(self, nodesToUpdate):
        '''This method is required defaulting to zero'''
        assert False, 'model should implement this function'
        return c_update(self.states, nodesToUpdate)
    cdef _updateState(self):
        return 0

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    def simulate(self, \
                 int nSamples = 100, int step = 1, \
                 dict pulse = {}, verbose = False):
        '''
        Returns state after sampling
        Input:
            : nSamples: number of required samples
            : step    : steps between samples
            : pulse   : a dict containing node to nudge at t - 1. Note this is the networkx node name not idx
        returns:
            :simulationResults: samples x states; note minimum samples is always 2
        '''

        cdef vector[int] names = [self.mapping[i] for i in pulse.keys()]
        cdef vector[double] nudges = list(pulse.values())
        cdef Pulse _pulse = [names, nudges]
        print(type(_pulse))
        self.test = _pulse
        print(type(self.test))
        print(_pulse.names)
        cdef long[  :] state  = self.states
        cdef str mode         = self.mode
        return c_simulate(state, nSamples, step, _pulse, mode, c_update)

ctypedef fused nameOrIdx:
    char
    int

ctypedef struct Pulse:
    vector[int] names
    vector[double] nudges
cdef np.ndarray c_update(long[:] x, long[:] y):
    return np.asarray(x)
@cython.wraparound(False)
@cython.boundscheck(False)
cdef np.ndarray c_sampleNodes(int nSamples, long[:] nodeIDs, str mode):
    cdef int length # length of target array
    # extract sample size from mode
    cdef int sampleSize
    length  = len(nodeIDs)
    if mode == 'single':
        sampleSize = 1
    else:
        sampleSize = length
    # allocate return matrix
    cdef long[:, :] samples = np.ndarray((nSamples, sampleSize), int)
    # sample
    for samplei in range(nSamples):
        samples[samplei]  = c_sample(nodeIDs, length, samplei, sampleSize)
    return samples

@cython.wraparound(False)
@cython.boundscheck(False)
cdef long[:] c_sample(long[:] nodeIDs, int length, int sampleN,\
              int sampleSize,\
              ):
    """
    Shuffles nodeID only when the current sample is larger
    than the shuffled array
    """
    cdef int start
    start = (sampleN * sampleSize) % length
    if start + sampleSize >= length:
        np.random.shuffle(nodeIDs)
    return nodeIDs[start : start + sampleSize]

@cython.wraparound(False)
@cython.boundscheck(False)
cdef np.ndarray c_updateState(long[:] state, long[:] nodesToUpdate, \
                    UPDATE funcPtr):

    return funcPtr(state, nodesToUpdate)

@cython.wraparound(False)
@cython.boundscheck(False)
cdef c_simulate(long[:] state, int nSamples, int step, Pulse pulse,\
                str mode, UPDATE funcPtr):
    cdef int N          = nSamples * step
    cdef int nNodes     = len(state)
    cdef long[:]nodeIDs = np.arange(nNodes)
    cdef long[:] newstate = state # maybe ridic
    cdef long[:, :] nodesToUpdate = c_sampleNodes(N, nodeIDs, mode)
    cdef simulationResults = np.zeros((nSamples + 1, nNodes))


    cdef int counter, sampleCounter = 1
    # start sampling
    simulationResults[0] = state
    for counter in range(N):
        state = c_updateState(state, nodesToUpdate[counter], funcPtr)
        if not counter % step:
            simulationResults[sampleCounter] = state
            sampleCounter += 1
    return simulationResults


#
#         # pre-define number of simulation steps
# #        nodesToUpdate = random.choice(self.nodeIDs, size = (nSamples * step + 1, self.nNodes if self.mode != 'single' else 1))        # this seems faster  ^ when no while loop...
#         if verbose:
#             print(\
#               'Simulation parameters\n number of samples: {}'\
#               '\n step size {}\n pulse : {}'.format(\
#                                                     nSamples, step, pulse\
#                                                     )\
#                             )
#         # this was first done inside update; pre-definition seems faster
#         # haven't explicitly tested the generator variant, but sense will reduce memory loads
#         # nodesToUpdate = \
#         # (self.sampleNodes[self.mode] for _ in range(nSamples * step + 1)) # convert to generator
#         cdef long[:, :] nodesToUpdate = self.sampleNodes(nSamples * step + 1)
#         # nodesToUpdate = np.array([self.sampleNodes[self.mode](self.nodeIDs) for i in range(nSamples * step + 1)])
#         # init storage vector
#         cdef simulationResults = np.zeros( (nSamples + 1, self.nNodes), dtype = self.statesDtype) # TODO: this should be a generator as well
#         cdef long[:] states               = self.states
#
#         if verbose : pbar = tqdm.tqdm(total = nSamples) # init progressbar
#         simulationResults[0, :]   = states # always store current state
#
#         # loop declaration
#         cdef double nudge
#         cdef int sampleCounter = 1
#         cdef int stepCounter   = 1 # zero step is already done?
#
#         cdef dict mapping     = self.mapping
#         cdef double[:] nudges = self.nudges
#         cdef updateState      = self.updateState
#         cdef str nudgeMode    = self.nudgeMode
#         while sampleCounter <= nSamples:
#             print(sampleCounter)
#             if pulse: # run pulse; remove nudges after sim step
#                 copyNudge = nudges.copy() # make deep copy  #TODO: think this will cause slowdowns for very large networks
#                 for node, nudge in pulse.items():
#                     if isinstance(node, tuple):
#                         for i in node:
#                             nudges[mapping[i]] = nudge
#                     else:
#                         # inject overwhelming pulse for 1 delta
#                         if nudgeMode == 'pulse':
#                             state = states[mapping[node]]
#                             nudges[mapping[node]] =  nudge
#                         # constant pulse add
#                         else:
#                             nudges[mapping[node]] = nudge
#
#             # self.updateState(next(nodesToUpdate))
#             updateState(nodesToUpdate[stepCounter])
#             # self.updateState(nodesToUpdate.__next__()) # update generator
#             # twice swap cuz otherway was annoying
#             if pulse:
#                 nudges = copyNudge # reset the copy
#                 if nudgeMode == 'pulse':
#                   pulse = {}
#                 elif nudgeMode == 'constant' and sampleCounter >= nSamples // 2:
#                   pulse = {}
#             if stepCounter % step == 0: # if stepCounter starts at zero 1 step is already made
#                 simulationResults[sampleCounter] = states
#                 sampleCounter += 1 # update sample counter
#                 if verbose: pbar.update(1)  # update PB
#             stepCounter += 1  # update step counter
#         # out of while
#         else:
#             return simulationResults
