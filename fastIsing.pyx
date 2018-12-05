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
                scipy,  functools, copy, time
from tqdm import tqdm

# ___CythonImports___
cimport cython
from cython cimport numeric
from cython.parallel cimport prange, parallel

cimport numpy as np # overwrite some c  backend from above

from libc.math cimport exp
from libc.stdlib cimport rand
from libcpp.map cimport map
from libcpp.vector cimport vector
from cython.operator cimport dereference, preincrement
from libc.stdio cimport printf

# use external exp
cdef extern from "vfastexp.h":
    double exp_approx "EXP" (double)
cdef extern from "limits.h":
    int INT_MAX
    int RAND_MAX

# cdef extern from "SFMT.h":
#     void init_gen_rand(int seed)
#     double genran_res53()

cdef extern from "stdlib.h":
    double drand48()
    void srand48(long int seedval)


from models cimport Model
cdef class Ising(Model)


# class implementation
# @cython.final # enforce extension type
@cython.auto_pickle(True)
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
        self.magSideOptions   = {'': 0, 'neg': -1, 'pos': 1}
        self.magSide          = magSide


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
    def t(self):
        return self._t

    @t.setter
    def t(self, value):
        self._t   = value
        self.beta = 1 / value if value != 0 else np.inf


    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.nonecheck(False)
    @cython.cdivision(True)
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
            long[::1] states
            long[:, ::1] r

        # print('Starting burnin')
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


    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.nonecheck(False)
    @cython.cdivision(True)
    cdef double energy(self, \
                       int  node, \
                       long[::1] states) :
        """
        input:
            :nsyncode: member of nodeIDs
        returns:
                :energy: current energy of systme config for node
        """
        cdef:
            long length            = self.adj[node].neighbors.size()
            long nodeState         = states[node]
            long neighbor, i
            double weight
            double energy          = 0
        for i in range(length):
            neighbor = self.adj[node].neighbors[i]
            weight   = self.adj[node].weights[i]
            energy  -= states[node] * states[neighbor] * weight + \
            self._H [neighbor] * states[neighbor]
        energy += self.__nudges[node]
        return energy

    cpdef long[::1] updateState(self, long[::1] nodesToUpdate):
        return self._updateState(nodesToUpdate)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.nonecheck(False)
    @cython.cdivision(True)
    cdef long[::1] _updateState(self, long[::1] nodesToUpdate) :
        """
        Determines the flip probability
        p = 1/(1 + exp(-beta * delta energy))
        """
        cdef:
            long[::1] states    = self._states # alias
            long[::1] newstates = self._newstates
            int  length = nodesToUpdate.shape[0]
            long node
            double Z = <double> self._nNodes
            double energy, p
        # for n in prange(length,  = True): # dont prange this
        for n in range(length):
            node      = nodesToUpdate[n]
            energy    = self.energy(node, states)
            p = 1 / ( 1. + exp(-self.beta * 2. * energy) )

            # if rand() / float(RAND_MAX) < p: # fast but not random
            # if self.sampler.sample() < p: # best option
            if np.random.rand()  < p: # slow but easy
                newstates[node] = -states[node]

        cdef double mu = 0 # MEAN
        cdef long NEG  = -1 # see the self.magSideOptions
        cdef long POS  = 1
        # printf('%d ', mu)
        # compute mean
        for node in range(self._nNodes):
            mu          += states[node] / Z
            states[node] = newstates[node] # update
            # check if conditions are met
        if (mu < 0 and self._magSide == POS) or\
         (mu > 0 and self._magSide == NEG):
            # printf('%f %d\n', mu, self._magSide)
            # flip if true
            for node in range(self._nNodes):
                states[node] = -states[node]
        return states



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

    cpdef  np.ndarray matchMagnetization(self,\
                              np.ndarray temps  = np.logspace(-3, 2, 20),\
                          int n             = int(1e3),\
                          int burninSamples = 100):
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
        cdef double tcopy   = self.t
        cdef results = np.zeros((2, temps.shape[0]))
        for idx, t in enumerate(temps):
            self.t          = t
            jdx             = self.magSideOptions[self.magSide]
            self.states     = jdx if jdx else 1 # rest to ones; only interested in how mag is kept
            # self.burnin(burninSamples)
            tmp             = self.simulate(n)
            results[0, idx] = abs(tmp.mean())
            results[1, idx] = ((tmp**2).mean() - tmp.mean()**2) * self.beta
        self.t = tcopy
        return results
