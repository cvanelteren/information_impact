# cython: infer_types=True
# distutils: language=c++
import numpy as np
cimport numpy as np
import networkx as nx, functools, time
from tqdm import tqdm

cimport cython
from cython.parallel cimport parallel, prange
# TODO: maybe add reversemapping [rmap] [done]
# TODO: insert the zero step in the simulate [done]
# TODO: make J weight to make model more general[done]
# TODO: the codes needs to written such that it is model independent; which it currently is\
#however the model does need to have a specific set of constraints [done]
#one of the current constraints is that it needs a base for the agent states
# TODO: add sync, async and single update methods [done]
DTYPE = np.int
ctypedef np.int DTYPE_T
# from libcpp.map cimport map as cmap
# from libcpp.string cimport string
# from libcpp.vector cimport vector
from libc.stdlib cimport rand

from cython.operator cimport dereference, preincrement
# from libc.stdlib cimport rand
from libc.string cimport strcmp
from libc.stdio cimport printf
from libcpp.vector cimport vector
from libcpp.map cimport map
from libcpp.unordered_map cimport unordered_map
# from RNG cimport RNG
cdef extern from "limits.h":
    int INT_MAX
    int RAND_MAX


cdef class Model
# from sampler cimport Sampler # mersenne sampler
# @cython.final
cdef class Model: # see pxd
    def __init__(self, \
                 graph, agentStates = [-1, 1], \
                 updateType         = 'single',\
                 nudgeType          = 'constant'):
        '''
        General class for the models
        It defines the expected methods for the model; this can be expanded
        to suite your personal needs but the methods defined here need are relied on
        by the rest of the package.

        It translates the networkx graph into numpy dependencies for speed.
        '''


        self.construct(graph, agentStates)
        self.nudgeType  = nudgeType
        self.updateType = updateType
        # self.sampler    = Sampler(42, 0., 1.)
        # self.sampler    = RNG(time.time())

    # TODO: make class pickable
    # hence the wrappers
    @property
    def states(self)    : return self._states
    @property
    def updateType(self): return self._updateType
    @property
    def nudgeType(self) : return self._nudgeType
    @property #return mem view of states
    def states(self)    : return self._states
    @property
    def nodeids(self)   : return self._nodeids
    @property
    def nudges(self)    : return self._nudges
    @property
    def nNodes(self)    : return self._nNodes
    @property
    def nStates(self)   : return self._nStates


    # TODO: reset all after new?
    @nudges.setter
    def nudges(self, vals):
        self._nudges[:] =  0
        for k, v in vals.items():
            idx = self.mapping[k]
            self._nudges[idx] = v

    @updateType.setter
    def updateType(self, value):
        assert value in 'sync async single serial'
        self._updateType = value
        # allow for mutation if async else independent updates
        if value == 'async':
            self._newstates = self._states
        else:
            if value == 'serial':
                self._nodeids = np.sort(self._nodeids) # enforce  for sampler
            self._newstates = self._states.copy()

    @nudgeType.setter
    def nudgeType(self, value):
        assert value in 'constant pulse'
        self._nudgeType = value

    @states.setter # TODO: expand
    def states(self, value):
        if isinstance(value, int):
            self._newstates[:] = value
            self._states   [:] = value
        elif isinstance(value, np.ndarray):
            self._newstates = value
            self._states    = value

    cpdef void construct(self, object graph, list agentStates):
        """
        Constructs adj matrix using structs
        """
        # check if graph has weights or states assigned and or nudges
        # note does not check all combinations
        # input validation / construct adj lists
        # defaults
        DEFAULTWEIGHT = 1.
        DEFAULTNUDGE  = 0.
        # DEFAULTSTATE  = random # don't use; just for clarity

        # forward declaration and init
        cdef:
            dict mapping = {} # made nodelabe to internal
            dict rmapping= {} # reverse
            str delim = '\t'
            np.ndarray states = np.zeros(graph.number_of_nodes(), int, 'C')
            int counter
            # double[::1] nudges = np.zeros(graph.number_of_nodes(), dtype = float)
            np.ndarray nudges = np.zeros(graph.number_of_nodes(), dtype = float)
            unordered_map[long, Connection] adj # see .pxd


        from ast import literal_eval
        connecting = graph.neighbors if isinstance(graph, nx.Graph) else nx.predecessors

        # cdef dict _neighbors = {}
        # cdef dict _weights   = {}
        # generate adjlist
        for line in nx.generate_multiline_adjlist(graph, delim):
            add = False # tmp for not overwriting doubles
            # input validation
            lineData = []
            # if second is not dict then it must be source
            for prop in line.split(delim):
                try:
                    i = literal_eval(prop) # throws error if only string
                    lineData.append(i)
                except:
                    lineData.append(prop) # for strings
            node, info = lineData
            # check properties, assign defaults
            if 'state' not in graph.node[node]:
                idx = np.random.choice(agentStates)
                graph.node[node]['state'] = idx
            if 'nudge' not in graph.node[node]:
                graph.node[node]['nudge'] =  DEFAULTNUDGE

            # if not dict then it is a source
            if isinstance(info, dict) is False:
                # add node to seen
                if node not in mapping:
                    # append to stack
                    counter             = len(mapping)
                    mapping[node]       = counter
                    rmapping[counter]   = node

                # set source
                source   = node
                sourceID = mapping[node]

                states[sourceID] = <long> graph.node[node]['state']
                nudges[sourceID] = <double> graph.node[node]['nudge']
            # check neighbors
            else:
                if 'weight' not in info:
                    graph[source][node]['weight'] = DEFAULTWEIGHT
                if node not in mapping:
                    counter           = len(mapping)
                    mapping[node]     = counter
                    rmapping[counter] = node

                # _neighbors[sourceID].push_back(mapping[node])
                # _weights[sourceID].push_back(graph[source][node]['weight'])
                # _weights[sourceID]   = _weights.get(sourceID, []) + [graph[source][node]['weight']]
                # _neighbors[sourceID] = _neighbors.get(sourceID, []) + [mapping[node]]
                # check if it has a reverse edge
                if graph.has_edge(node, source):
                    sincID = mapping[node]
                    weight = graph[node][source]['weight']
                    # check if t he node is already in stack
                    if sourceID in set(adj[sincID]) :
                        add = True
                            # _weights[tmp]   = _weights.get(tmp, []) + [weight]
                            # _neighbors[sincID] = _neighbors.get(sincID, []) + [sourceID]
                    # not found so we should add
                    else:
                        add = True
                # add source > node
                sincID = <long> mapping[node]
                adj[sourceID].neighbors.push_back(<long> mapping[node])
                adj[sourceID].weights.push_back(<double> graph[source][node]['weight'])
                # add reverse
                if add:
                    adj[sincID].neighbors.push_back( <long> sourceID)
                    adj[sincID].weights.push_back( <double> graph[node][source]['weight'])

        # public and python accessible
        self.graph    = graph
        self.mapping  = mapping
        self.rmapping = rmapping

        self.adj = adj

        self.agentStates = np.asarray(agentStates, dtype = int)

        self._nudges = nudges
        self._nStates = len(agentStates)


        #private
        # note nodeids will be shuffled and cannot be trusted for mapping
        # use mapping to get the correct state for the nodes
        _nodeids = np.arange(graph.number_of_nodes(), dtype = long)
        self._nodeids      = _nodeids
        self._states       = states
        self._newstates    = states.copy()
        self._nNodes = graph.number_of_nodes()

    cdef long[::1]  _updateState(self, long[::1] nodesToUpdate) nogil:
        return self._nodeids


    cpdef long[::1] updateState(self, long[::1] nodesToUpdate):
        return self._nodeids

    # property _states:
    #     def __get__(self):
    #         return self._states.base
    #     def __set__(self, values):
    #         cdef long[::1] self._states

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.nonecheck(False)
    @cython.cdivision(True)
    cdef long [:, ::1] sampleNodes(self, long  nSamples) nogil:
        """
        Shuffles nodeID only when the current sample is larger
        than the shuffled array

        """
        # check the amount of samples to get
        cdef int sampleSize
        if self._updateType == 'single':
            sampleSize = 1
        # elif self._updateType == 'serial':
        #     return self._nodeids
        else:
            sampleSize = self._nNodes

        cdef:
            # TODO replace this with a nogil version
            long [:, ::1] samples
            long sample
            long start
            long i, j, k
            long samplei
        with gil:
            samples = np.ndarray((nSamples, sampleSize), dtype = int)
        # TODO: single updates of size one won't get shuffled
        for samplei in range(nSamples):
            # shuffle if the current tracker is larger than the array
            start = (samplei * sampleSize) % self._nNodes
            if start + sampleSize >= self._nNodes:
                for i in range(self._nNodes): # TODO: replace this with new samplers
                    # j = <long> i + dist(gen) * (self._nNodes - i) + 1
                    j                = <long> (i + rand() / (RAND_MAX / (self._nNodes - i) + 1) )
                    k                = self._nodeids[j]
                    self._nodeids[j] = self._nodeids[i]
                    self._nodeids[i] = k
            # assign the samples
            for j in range(sampleSize):
                samples[samplei, j] = self._nodeids[start + j]
                # samples[samplei] = nodeIDs[start : start + sampleSize]
        return samples



    cpdef void reset(self):
        self.states = np.random.choice(\
                self.agentStates, size = self._nNodes)


    def removeAllNudges(self):
        """
        Sets all nudges to zero
        """
        self.nudges[:] = 0

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.nonecheck(False)
    @cython.cdivision(True)
    cpdef simulate(self, long long int  samples):
        cdef:
            long[:, ::1] results = np.zeros((samples, self._nNodes), int)
            long[:, ::1] r = self.sampleNodes(samples)
            int i
        for i in range(samples):
            results[i] = self.updateState(r[i])
        return results.base # convert back to normal arraay


    # TODOL move this back ^
    # cdef long[::1] updateState(self, int[:] nodesToUpdate):
    #     ""
    #     Implement this method
    #     """
    #     assert False
    # @cython.wraparound(False)
    # @cython.boundscheck(False)
    # @cython.nonecheck(False)
    # cpdef simulate(self, int nSamples, int step):
    #     # pre-define number of simulation steps
    #     cdef int N   = nSamples * step + 1
    #     cdef long[:, :] nodesToUpdate = self.sampleNodes(N)
    #     # nodesToUpdate = np.array([self.sampleNodes[self.mode](self.nodeIDs) for i in range(nSamples * step + 1)])
    #     # init storage vector
    #     cdef simulationResults = np.zeros( (nSamples + 1, self.nNodes), dtype = self.statesDtype) # TODO: this should be a generator as well
    #     cdef long[:] states               = self.states
    #
    #     simulationResults[0, :]   = states # always store current state
    #
    #     # loop declaration
    #     cdef double nudge
    #     cdef int sampleCounter = 1
    #     cdef int stepCounter   = 1 # zero step is already done?
    #
    #     cdef dict mapping     = self.mapping
    #     cdef double[:] nudges = self.nudges
    #     cdef updateState      = self.updateState
    #     cdef str nudgeType    = self.nudgeType
    #     for sampleCounter in range(N):
    #         if pulse: # run pulse; remove nudges after sim step
    #             copyNudge = nudges.copy() # make deep copy  #TODO: think this will cause slowdowns for very large networks
    #             for node, nudge in pulse.items():
    #                 if isinstance(node, tuple):
    #                     for i in node:
    #                         nudges[mapping[i]] = nudge
    #                 else:
    #                     # inject overwhelming pulse for 1 delta
    #                     if nudgeType == 'pulse':
    #                         state = states[mapping[node]]
    #                         nudges[mapping[node]] =  nudge
    #                     # constant pulse add
    #                     else:
    #                         nudges[mapping[node]] = nudge
    #
    #         # self.updateState(next(nodesToUpdate))
    #         updateState(nodesToUpdate[stepCounter])
    #         # self.updateState(nodesToUpdate.__next__()) # update generator
    #         # twice swap cuz otherway was annoying
    #         if pulse:
    #             nudges = copyNudge # reset the copy
    #             if nudgeType == 'pulse':
    #               pulse = {}
    #             elif nudgeType == 'constant' and sampleCounter >= nSamples // 2:
    #               pulse = {}
    #         if stepCounter % step == 0: # if stepCounter starts at zero 1 step is already made
    #             simulationResults[sampleCounter] = states
    #             sampleCounter += 1 # update sample counter
    #         stepCounter += 1  # update step counter
    #     # out of while
    #     else:
    #         return simulationResults
