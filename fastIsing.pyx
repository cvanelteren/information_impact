#!/usr/bin/env python3
# distutils: language=c++
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  6 09:36:17 2018

@author: Casper van Elteren
"""
# cython: infer_types=True
import numpy  as np

from scipy.stats import linregress
import networkx as nx, multiprocessing as mp, \
                tqdm, scipy,  functools, copy

# ___CythonImports___
cimport cython
from cython cimport numeric
from cython.parallel cimport prange, parallel

from cython.operator cimport dereference, preincrement
cimport numpy as np # overwrite some c  backend from above

from libc.math cimport exp
from libc.stdlib cimport rand

from libcpp.map cimport map
from libcpp.vector cimport vector
from cython.operator cimport dereference, preincrement
# use external exp
cdef extern from "vfastexp.h":
    double exp_approx "EXP" (double)
cdef extern from "limits.h":
    int INT_MAX

from models cimport Model

cdef class Ising(Model)

# class implementation
@cython.final # enforce extension type
cdef class Ising(Model):
    def __init__(self, \
                 graph,\
                 temperature = 1,\
                 agentStates = [-1 ,1],\
                 nudgeType   = 'constant',\
                 updateType  = 'async', \
                 magSide     = 'neg',\
                 ):

        super(Ising, self).__init__(\
                  graph       = graph, \
                  agentStates = agentStates, \
                  updateType  = updateType, \
                  nudgeType   = nudgeType)


        cdef np.ndarray H  = np.zeros(self.graph.number_of_nodes(), float)
        for node, nodeID in self.mapping.items():
            H[nodeID] = graph.nodes()[node].get('H', 0)

        # specific model parameters
        self._H               = H
        self.beta             = np.inf if temperature == 0 else 1 / temperature
        self.t                = temperature
        self.magSide          = magSide
    @property
    def H(self): return self._H

    # added property cuz im lazy
    @property
    def t(self):
        return self._t
    @t.setter
    def t(self, value):
        self._t = value
        self.beta = 1 / value if value != 0 else np.inf

    cpdef np.ndarray burnin(self,\
                 int samples = int(1e2), double threshold = 1e-2, ):
        """
        Go to equilibrium distribution; uses magnetization and linear regression
        to see if the system is stabel
        """

        # magnetization function
        magnetization = np.mean
        cdef:
            y  = np.array(magnetization(self.states)) # for regression
            int h, counter = 0 # tmp var and counter
            double beta        # slope value
            np.ndarray x # for regression
            long[:] states
            long[:, :] r

        print('Starting burnin')
        while True:
            r      = self.sampleNodes(1) # produce shuffle
            states = self.updateState(r[0]) # update state
            # check if magnetization = constant
            y      = np.hstack((y, np.abs(magnetization(states))))
            if counter > samples :
                # do linear regression
                h = len(y + 2) # plus 2 due to starting point
                x = np.arange(h)
                beta = linregress(x, y).slope
                if abs(beta) < threshold:
                    break
            counter += 1
        else:
            print('Number of bunin samples used {0}\n\n'.format(counter))
            print(f'absolute mean magnetization last sample {abs(y[-1])}')
        return y

    # cpdef reset(self, useAtleastNSamples = None):
    #     """Resets the states to random"""
    #     self.__states = np.random.choice(self.agentStates, size = self.nNodes)
    #     if useAtleastNSamples is not None:
    #         self.burnin(useAtleastNSamples)


    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.nonecheck(False)
    @cython.cdivision(True)
    cdef double energy(self, \
                       int  node, \
                       long[:] index,\
                       double [:] weights,\
                       double nudge,\
                       long[:] states):
        """
        input:
            :node: member of nodeIDs
        returns:
                :energy: current energy of systme config for node
                :flipEnergy: energy if node flips state
        """
        cdef:
            double[::1] H    = self._H  # acces the c property
            long length = index.shape[0]
            # index  = np.asarray(indices, dtype = int)
            double energy = 0.
            int neighbor, i
            double weight, Hi
            cdef long nodeState = states[node]
            long neighborState

        # compute hamiltonian and add possible nudge
        # note weights is a matrix of shape (1, nNodes)
        # with nogil, parallel():
        # with nogil:
        for i in range(length):
            energy -=  nodeState * states[index[i]] * weights[i] \
            + H[index[i]] * states[index[i]]
        energy += nudge
        return energy

    # @cython.boundscheck(False)
    # @cython.wraparound(False)
    # @cython.nonecheck(False)
    # cpdef updateState(self, long[:] nodesToUpdate):
    #     return self._updateState(nodesToUpdate)
    cpdef long[::1] updateState(self, long[:] nodesToUpdate):
        return self._updateState(nodesToUpdate)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.nonecheck(False)
    @cython.cdivision(True)
    cdef long[::1] _updateState(self, long[:] nodesToUpdate):
        """
        Determines the flip probability
        p = 1/(1 + exp(-beta * delta energy))
        """
        cdef:
            long[::1] states = self._states # alias
            long [::1] newstates #forward declaration
            int  length = nodesToUpdate.shape[0]
            str magSide = self.magSide
            int  n
            long node
            double energy, p
            double beta        = self.beta
            double[::1] nudges = self.__nudges
            dict weights       = self.weights
            dict neighbors     = self.neighbors
        # cdef map[int, vector[int]] w = weights
        # cdef map[int, vector[int]] a = neighbors
        #
        # cdef map[int, vector[int]].iterator weightsb = w.begin()
        # cdef map[int, vector[int]].iterator weightse = w.end()
        #
        # cdef map[int, vector[int]].iterator neighborssb = a.begin()
        # cdef map[int, vector[int]].iterator neighborse = a.end()
        # # allow mutability if async; single doesn't matter
        # cdef int total = 0
        # cdef int j, i
        # with nogil:
        #     while neighborssb != neighborse:
        #         j = dereference(neighborssb).second.size()
        #         for i in range(j):
        #             total += dereference(neighborssb).second[i]
        #         preincrement(neighborssb)
        # print(total)
        if self.__updateType == 'async':
            newstates = states
        else:
            newstates = states.copy()
        # for n in prange(length, nogil = True):
        for n in range(length):
            node      = nodesToUpdate[n]


            # nudge     = nudges[node]
            # weight    = weights[node]
            # neighbor  = neighbors[node]
            #  TODO: dunno how to typecast this; fused types can be used but dunno how
            # weights = self.adj[:, node].data
            # indices = self.adj[:, node].indices
            # print(node, np.asarray(weights), np.asarray(indices))
            # check if node has input

            # energy = self.energy(node, neighbor, weight,\
            #                      nudge, newstates)
            energy = self.energy(\
                                 node, neighbors[node], weights[node],\
                                 nudges[node], newstates)
            p = 1 / ( 1. + exp(-beta * 2. * energy) )

            if rand() / float(INT_MAX) < p: # faster
                states[node] = -newstates[node]

        cdef double mu = 0
        for n in range(self._nNodes):
            mu += states[n] / float(self._nNodes)
        if (mu < 0 and magSide == 'pos') or (mu > 0 and magSide == 'neg'):
            for n in range(self._nNodes):
                states[n] = -states[n]
        return states

    cpdef simulate(self, long samples):
        cdef:
            long[:, ::1] results = np.zeros((samples, self.graph.number_of_nodes()), int)
            long[:, :] r = self.sampleNodes(samples)
            int i
        for i in range(samples):
            results[i] = self.updateState(r[i])
        return results.base

    # cpdef computeProb(self):
    #     """
    #     Compute the node probability for the current state p_i = 1/z * (1 + exp( -beta * energy))**-1
    #     """
    #
    #     probs = np.zeros(self.nNodes)
    #     for node in self.nodeIDs:
    #         en = self.energy(node, self.states[node])
    #         probs[node] = exp(-self.beta * en)
    #     return probs / np.nansum(probs)

    def matchMagnetization(self,\
                          temps = np.logspace(-3, 2, 20),\
                          n = int(1e3),\
                          burninSamples = 100):

        """
        Computes the magnetization as a function of temperatures
        Input:
              :temps: a range of temperatures
              :n:     number of samples to simulate for
              :burninSamples: number of samples to throw away before sampling
        Returns:
              :temps: the temperature range as input
              :mag:  the magnetization for t in temps
              :sus:  the magnetic susceptibility
        """
        # start from 1s and check how likely nodes will flip
        # cdef double[:] H  = np.zeros( len(temps) )   # magnetization
        # cdef double[:] HH = np.zeros( ( len(temps ))) # susceptibility

        func = functools.partial(matchMagnetization, nSamples = n,
                    burninSamples = burninSamples)
        x = []
        for t in temps:
            new   = copy.copy(self)
            x.append((new, t))

            with mp.Pool(processes = mp.cpu_count()) as p:
                results = np.array(p.map(func, x))
         # results = np.zeros((temps.size, 2))
         # for idx, val in enumerate(x):
         #     print(val)
         #     results[idx, :] = func(val)
        return results.T


    def removeAllNudges(self):
        """
        Sets all nudges to zero
        """
        self.nudges[:] = 0


def matchMagnetization(modelt, nSamples, burninSamples):
    """
    compute in parallel the magnetization
    """
    model, t = modelt
    model.t = t
    model.states.fill(-1) # rest to ones; only interested in how mag is kept
    # self.reset()
    model.burnin(burninSamples)
    res = np.asarray(model.simulate(nSamples))
    H   = abs(res.mean())
    HH =  ((res**2).mean() - res.mean()**2) * model.beta # susceptibility
    return H, HH
