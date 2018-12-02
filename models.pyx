# cython: infer_types=True
# distutils: language=c
import numpy as np
cimport numpy as np
import networkx as nx, tqdm, functools

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
from libc.stdlib cimport rand

from libc.stdio cimport printf
cdef extern from "limits.h":
    double INT_MAX
# ctypedef np.ndarray (*UPDATE)(long[:] state, long[:] nodesToUpdate)
# forward declaration
cdef class Model


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


        # graph      = kwargs['graph']
        # agentStates= kwargs['agentStates']
        self.nudgeType  = nudgeType
        self.updateType = updateType

        # construct other properties

        self.construct(graph, agentStates)

    @property
    def states(self): return self._states
    @property
    def updateType(self): return self.__updateType
    @property
    def nudgeType(self): return self.__nudgeType
    @property #return mem view of states
    def states(self): return self._states
    @property
    def nodeids(self): return self._nodeids


    @updateType.setter
    def updateType(self, value):
        assert value in 'sync async single serial'
        self.__updateType = value
    @nudgeType.setter
    def nudgeType(self, value):
        assert value in 'constant pulse'
        self.__nudgeType = value
    @states.setter # TODO: expand
    def states(self, value):
        if isinstance(value, np.ndarray):
            self._states = value
        else:
            self._states[:] = value


    cpdef void construct(self, object graph, list agentStates):
        """
        Constructs adj matrix using structs
        """
        # check if graph has weights or states assigned and or nudges
        # note does not check all combinations
        # input validation / construct adj lists
        cdef double[:] bins = np.linspace(0, 1, len(agentStates) + 1)
        # defaults
        DEFAULTWEIGHT = 1.
        DEFAULTNUDGE  = 0.
        # DEFAULTSTATE  = random # don't use; just for clarity

        # forward declaration and init
        cdef:
            dict mapping = {}
            dict rmapping= {}
            str delim = '\t'
            long[::1] states = np.zeros(graph.number_of_nodes(), int, 'C')
            int counter
            double[::1] nudges = np.zeros(graph.number_of_nodes(), float, 'C')

        # check all the possible edges
        from ast import literal_eval
        connecting = graph.neighbors if isinstance(graph, nx.Graph) else nx.predecessors

        cdef dict neighbors = {}
        cdef dict weights   = {}
        # generate adjlist
        for line in nx.generate_multiline_adjlist(graph, delim):
            # input validation
            tmp = []
            # if second is not dict then it must be source
            for prop in line.split(delim):
                try:
                    i = literal_eval(prop) # throws error if only string
                    tmp.append(i)
                except:
                    tmp.append(prop) # for strings
            node, info = tmp
            # check properties, assign defaults
            if 'state' not in graph.node[node]:
                idx = np.digitize(np.random.random_sample(), bins = bins)
                graph.node[node]['state'] = agentStates[idx - 1]
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

                states[sourceID] = graph.node[node]['state']
                nudges[sourceID] = graph.node[node]['nudge']
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
                weights[sourceID]   = weights.get(sourceID, []) + [graph[source][node]['weight']]
                neighbors[sourceID] = neighbors.get(sourceID, []) + [mapping[node]]
                # check if it has a reverse edge
                if graph.has_edge(node, source):
                    tmp            = mapping[node]
                    weight         = graph[node][source]['weight']
                    if tmp in neighbors:
                        if source not in neighbors[tmp]:
                            weights[tmp]   = weights.get(tmp, []) + [weight]
                            neighbors[tmp] = neighbors.get(tmp, []) + [sourceID]
                    else:
                        weights[tmp]   = weights.get(tmp, []) + [weight]
                        neighbors[tmp] = neighbors.get(tmp, []) + [sourceID]


        # TODO: copy in place
        weights        = {k : np.array(v) for k, v in weights.items()}
        neighbors      = {k : np.array(v, dtype = int) for k, v in neighbors.items()}
        self.neighbors = neighbors
        self.weights   = weights
        # public and python accessible
        self.graph    = graph
        self.mapping  = mapping
        self.rmapping = rmapping

        # self.adj.data.astype(np.float64) #TODO: this doesn't work..

        # enforce dtypes: reduce this if needed

        self.agentStates = np.asarray(agentStates, dtype = int)

        self.__nudges = nudges
        self._nStates = len(agentStates)
        self.nStates  = len(agentStates)

        #private
        # note nodeids will be shuffled and cannot be trusted for mapping
        # use mapping to get the correct state for the nodes
        cdef np.ndarray _nodeids = np.arange(graph.number_of_nodes(), dtype = long)
        self._nodeids = _nodeids
        self._states = states
        self._nNodes = graph.number_of_nodes()
    @cython.wraparound(False)
    @cython.boundscheck(False)
    @cython.nonecheck(False)
    cpdef long [:, ::1] sampleNodes(self, long nSamples):
        """
            Python accessible function to sample nodes
        """
        cdef:

            # initialization
            nodeIDs           = self._nodeids
            long length       = self._nNodes # length of target array
            updateType        = self.__updateType

        cdef int sampleSize
        if updateType == 'single':
            sampleSize = 1
        elif updateType == 'serial':
            sampleSize = 0
        else:
            sampleSize = length
        return self.c_sample(nodeIDs, length,  nSamples, sampleSize)

    @cython.wraparound(False)
    @cython.boundscheck(False)
    @cython.nonecheck(False)
    @cython.cdivision(True)
    cdef long[:, ::1] c_sample(self,
                    long[::1] nodeIDs, \
                    int length, int nSamples,\
                    int sampleSize,\
                    ) :
        """
        Shuffles nodeID only when the current sample is larger
        than the shuffled array
        """
        cdef long [:, ::1] samples = np.ndarray((nSamples, sampleSize), dtype = int)
        cdef:
            long sample
            long start
            long i, j, k
            long samplei
        with nogil:
            for samplei in range(nSamples):
                start = (samplei * sampleSize) % length
                if start + sampleSize >= length:
                    # np.random.shuffle(nodeIDs)
                    for i in range(length):
                        j = <long> (i + rand() / (INT_MAX / (length - i)) )
                        # printf('%d\n',j)
                        k = nodeIDs[j]
                        nodeIDs[j] = nodeIDs[i]
                        nodeIDs[i] = k

                for j in range(sampleSize):
                    samples[samplei, j] = nodeIDs[start + j]
                    # samples[samplei] = nodeIDs[start : start + sampleSize]
        return samples

    cpdef void reset(self):
        self.states = np.random.choice(self.agentStates, size = self._nNodes)
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
