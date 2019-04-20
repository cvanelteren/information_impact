# cython: infer_types=True
# distutils: language=c++
from Models.models cimport Model
from libcpp.vector cimport vector

# from models cimport Model
import numpy  as np
cimport numpy as np

from libc.math cimport exp
cimport cython
cdef class Potts(Model):
    def __init__(self, \
                 graph,\
                 temperature = 1,\
                 agentStates = [-1 ,1, 0],\
                 nudgeType   = 'constant',\
                 updateType  = 'async', \
                 ):
        super(Potts, self).__init__(\
                  graph           = graph, \
                  agentStates  = agentStates, \
                  updateType  = updateType, \
                  nudgeType   = nudgeType)


        cdef np.ndarray H  = np.zeros(self.graph.number_of_nodes(), float)
        for node, nodeID in self.mapping.items():
            H[nodeID] = graph.nodes()[node].get('H', 0)
        # for some reason deepcopy works with this enabled...
        self.states           = np.asarray(self.states.base).copy()
        self.nudges         = np.asarray(self.nudges.base).copy()
        # specific model parameters
        self._H               = H
        # self._beta             = np.inf if temperature == 0 else 1 / temperature
        self.t                  = temperature

    @property
    def magSide(self):
        for k, v in self.magSideOptions.items():
            if v == self._magSide:
                return k
    @magSide.setter
    def magSide(self, value):
        idx = self.magSideOptions.get(value,\
              f'Option not recognized. Options {self.magSideOptions.keys()}')
        if isinstance(idx, int):
            self._magSide = idx
        else:
            print(idx)
    @property
    def H(self): return self._H

    @property
    def beta(self): return self._beta

    @beta.setter
    def beta(self, value):
        self._beta = value

    @property
    def t(self):
        return self._t

    @t.setter
    def t(self, value):
        self._t   = value
        self.beta = 1 / value if value != 0 else np.inf

    cpdef long[::1] updateState(self, long[::1] nodesToUpdate):
        return self._updateState(nodesToUpdate)

    cpdef vector[double] siteEnergy(self, long[::1] states):
        cdef:
            vector[double] siteEnergy
            int node
            double Z
        for node in range(self._nNodes):
            Z = self._adj[node].neighbors.size()
            siteEnergy.push_back(self.energy(node, states)[0] / Z)
        return siteEnergy


    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.nonecheck(False)
    @cython.cdivision(True)
    @cython.initializedcheck(False)
    @cython.overflowcheck(False)
    cdef vector[double] energy(self, int node, long[::1] states) nogil:
        cdef:
            long neighbors = self._adj[node].neighbors.size()
            long neighbor, neighboridx
            double weight
            long possibleState
            vector[double] energy
        # fill buffer
        # TODO: change this to more efficient buffer
        # keep track of:
        #   - energy of current state
        #   - energy of possible state
        #   - the possible state
        for possibleState in range(3):
            energy.push_back(0)
        # count the neighbors in the different possible states

        # draw random new state
        cdef int testState = <int> (self.rand() * self._nStates)
        testState = self.agentStates[testState]

        energy[0] = self._H[node]
        energy[1] = self._H[node]
        energy[2] = testState # keep track of possible new state
        for neighboridx in range(neighbors):
            neighbor   = self._adj[node].neighbors[neighboridx]
            weight     = self._adj[node].weights[neighboridx]
            if states[neighbor] == states[node]:
                energy[0] -= weight
            if states[neighbor] == testState:
                energy[1] -= weight
        # with gil: print(energy)
        return energy
    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.nonecheck(False)
    @cython.cdivision(True)
    @cython.initializedcheck(False)
    @cython.overflowcheck(False)
    cdef long[::1] _updateState(self, long[::1] nodesToUpdate) nogil:

        """
        Generate conditional distribution based on a change in state
        For all agent states compute the likelihood of staying in that state
        """

        cdef:
            int nodes = nodesToUpdate.shape[0]
            long node, nodeidx
            vector[double] probs
            int agentState
            double randomNumber
        for nodeidx in range(nodes):
            node         = nodesToUpdate[nodeidx]
            probs        = self.energy(node, self._states)
            randomNumber = self.rand()
            # with gil:
                # print(probs)
            if randomNumber <= exp(- self._beta * (probs[1] - probs[0])):
                self._newstates[node] = <int> probs[2]
        # repopulate buffer
        for node in range(self._nNodes):
            self._states[node] = self._newstates[node]
        return self._states
