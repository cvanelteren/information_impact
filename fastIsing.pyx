#!/usr/bin/env python3
# distutils: language=c
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

# use external exp
cdef extern from "vfastexp.h":
    double exp_approx "EXP" (double)
cdef extern from "limits.h":
    int INT_MAX

from models cimport Model
# TODO: this is still too pythonic. The conversion was made from direct python code; future me needs to hack this into c
# forward declaration

# cdef fused longdouble:
#     long
#     double

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
        self.beta = 1/value if value != 0 else np.inf

    cpdef np.ndarray burnin(self,\
                 int samples = int(1e2), double threshold = 1e-2, ):
        '''
        Go to equilibrium distribution; uses magnetization and linear regression
        to see if the system is stabel
        '''

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
    #     '''Resets the states to random'''
    #     self.__states = np.random.choice(self.agentStates, size = self.nNodes)
    #     if useAtleastNSamples is not None:
    #         self.burnin(useAtleastNSamples)


    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.nonecheck(False)
    @cython.cdivision(True)
    cdef double energy(self, \
                       int node, \
                       int[:] index,\
                       double [:] weights):
        '''
        input:
            :node: member of nodeIDs
        returns:
                :energy: current energy of systme config for node
                :flipEnergy: energy if node flips state
        '''
        cdef:
            long[::1] states = self.__states
            double[::1] H    = self._H  # acces the c property
            long length = len(index)
            # index  = np.asarray(indices, dtype = int)
            double energy = 0.
            int neighbor, i
            double weight, Hi
            cdef double nudge = self.__nudges[node]
            cdef long nodeState = states[node]
            long neighborState


        # compute hamiltonian and add possible nudge
        # note weights is a matrix of shape (1, nNodes)
        # with nogil, parallel():
        with nogil, parallel():
            for i in prange(length):
            # for i in prange(length, schedule = 'static'):
                # neighbor      = index[i]
                # neighborState = states[neighbor]
                # weight        = weights[neighbor]
                # Hi            = H[neighbor]
                # energy       += weight * neighborState
                # energy       += neighborState * Hi
                energy += states[index[i]] * weights[index[i]] + H[index[i]] * states[index[i]]
        energy *= nodeState
        energy += nudge
        return -energy

    # @cython.boundscheck(False)
    # @cython.wraparound(False)
    # @cython.nonecheck(False)
    # cpdef updateState(self, long[:] nodesToUpdate):
    #     return self._updateState(nodesToUpdate)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.nonecheck(False)
    @cython.cdivision(True)
    cpdef long[::1] updateState(self, long [:] nodesToUpdate):
        '''
        Determines the flip probability
        p = 1/(1 + exp(-beta * delta energy))
        '''
        cdef:
            states = self.__states # alias
            long [::1] newstates #forward declaration
            int  length = len(nodesToUpdate)
            str magSide = self.magSide
            int  n
            long node
            double energy, p
            int[::1] indices
            double[:] weights
            cdef double beta = self.beta

        # allow mutability if async; single doesn't matter
        if self.__updateType == 'async':
            newstates = states
        else:
            newstates = states.copy()

        # for n in prange(length, nogil = True):
        for n in range(length):
            node    = nodesToUpdate[n]
            #  TODO: dunno how to typecast this; fused types can be used but dunno how
            # weights = self.adj[:, node].data
            # indices = self.adj[:, node].indices
            # print(node, np.asarray(weights), np.asarray(indices))
            # check if node has input
            if self.index[node].shape[0] != 0:
                energy = self.energy(node, self.index[node], self.adj[:, node])
                # print(energy)
                p = 1 / ( 1 + exp(- beta * 2. * energy) )
            else:
                p = .5
            if rand() / float(INT_MAX) < p: # faster
                newstates[node] = -states[node]

        mu = np.mean(newstates)
        if (mu < 0 and magSide == 'pos') or (mu > 0 and magSide == 'neg'):
            # print('inverting', mu, magSide)
            for i in range(length):
                newstates[i] = -newstates[i]
        states = newstates # possible overwrite
        return states #
    cpdef np.ndarray simulate(self, int samples):
        cdef:
            long[:, ::1] results = np.zeros((samples, self.graph.number_of_nodes()), int)
            long[:, ::1] r = self.sampleNodes(samples)
            int i
        for i in range(samples):
            results[i] = self.updateState(r[i])

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
    #
    # def matchMagnetization(self,\
    #                       temps = np.logspace(-3, 2, 20),\
    #                       n = int(1e3),\
    #                       burninSamples = 100):
    #  """
    #  Computes the magnetization as a function of temperatures
    #  Input:
    #     :temps: a range of temperatures
    #     :n:     number of samples to simulate for
    #     :burninSamples: number of samples to throw away before sampling
    # Returns:
    #     :temps: the temperature range as input
    #     :mag:  the magnetization for t in temps
    #     :sus:  the magnetic susceptibility
    #  """
    #  # start from 1s and check how likely nodes will flip
    #  # cdef double[:] H  = np.zeros( len(temps) )   # magnetization
    #  # cdef double[:] HH = np.zeros( ( len(temps ))) # susceptibility
    #
    #  func = functools.partial(matchMagnetization, nSamples = n,
    #                burninSamples = burninSamples)
    #  x = []
    #  for t in temps:
    #      new   = copy.copy(self)
    #      x.append((new, t))
    #
    #  with mp.Pool(processes = mp.cpu_count()) as p:
    #    results = np.array(p.map(func, x))
    #  # results = np.zeros((temps.size, 2))
    #  # for idx, val in enumerate(x):
    #  #     print(val)
    #  #     results[idx, :] = func(val)
    #  H, HH = results.T
    #
    #  return H, HH

    # def fit(self, temps, magRange, nSamples,
    #         burninSamples, func,
    #         fit_func):
    #   """
    #   Fits a kernel to :func: output for different temperatures.
    #   It is aimed to provide an indication of the influence of temperature
    #   for a particular target func
    #   """
    #
    #   y    = func(temps, nSamples, burninSamples)
    #   a, b = scipy.optimize.curve_fit(func, temps, y)
    #
    #   def f_root(x, opts, b):
    #     return func(x, *opts, b)
    #   roots = []


    #
    # def matchingEntropy(self, targetEntropy = 1/np.e, \
    #                    temps = np.logspace(-3, 1, 20), n = 10,\
    #                    ):
    #   """
    #   Matches the temperature to :targetEntropy:. This function is deprecated
    #   """
    #   def func(x, a, b, c, d):
    #     return   a / (1 + np.exp(b * (x - c)) ) + d
    #   def ifunc(x, a, b, c, d):
    #     return   c + 1/b * np.log(a / (x - d) - 1)
    #   H = np.zeros(len(temps))
    #   self.reset()
    #   energy = np.array([[self.energy(node, s) for s in self.agentStates] for node in self.nodeIDs])
    #   for tidx, t in enumerate(temps):
    #     self.t = t
    #     probs   = 1 / (1 + np.exp(self.beta * energy))
    #     H[tidx] = np.mean(-np.nansum(probs * np.log2(probs), axis = 1))
    #   try:
    #       a, b = scipy.optimize.curve_fit(func, temps, H,  maxfev = 10000)
    #       matchedEntropy = ifunc(targetEntropy, *a)
    #       return temps, H, matchedEntropy
    #   except Exception as e:
    #         print(e)
    #         return temps, H
    #
    #
    # def removeAllNudges(self):
    #     '''
    #     Sets all nudges to zero
    #     '''
    #     self.nudges = np.zeros(self.nudges.shape)
# @cython.boundscheck(False)
# cdef np.ndarray c_updateState(int [:] nodesToUpdate,\
# int [:] states,\
# double [:] interactions,\
# int [:] edgeData,\
# double beta,\
# double [:] H,\
# str magSide,\
# str updateType):
#     '''
#     Determines the flip probability
#     p = 1/(1 + exp(-beta * delta energy))
#     '''
#
#     states = states.copy() if updateType== 'sync' else states # only copy if sync else alias
#     cdef long N = len(nodesToUpdate)
#     cdef int n, node
#     cdef double energy
#     cdef interaction
#     cdef edgeDatum
#
#     for n in range(N):
#       node        = nodesToUpdate[n]
#       edgeDatum   = edgeData[node]
#       interaction = interactions[node]
#       energy      = c_energy(node, states, edgeDatum, interaction, H)
#       # TODO: change this mess
#       # # heatbath glauber
#       # if self.updateMethod == 'glauber':
#       tmp = -beta *  2 * energy
#       # tmp = - self.beta * energy * 2
#       tmp = 0 if np.isnan(tmp) else tmp
#       p = 1 / ( 1 + exp(tmp) )
#       # if rand() / float(INT_MAX) < p:
#       # print(p, energy)
#       if rand() / float(INT_MAX)  <= p:
#         states[node] = -states[node]
#     return states

#
#
# def fitTemperature(temperature, graph, nSamples, step, fitModel = Ising):
#     '''
#     Used by match temperature to match the entropy
#     for a specific graph of specific size
#     '''
#     model = fitModel(graph = graph, temperature = temperature, doBurnin = True)
#     return information.getSnapShots(model = model, nSamples = nSamples,\
#                                     step = step)[:2] # only keep the node probs
#
#
# def matchMagnetization(modelt, nSamples, burninSamples):
#   '''
#   compute in parallel the magnetization
#   '''
#   model, t = modelt
#   model.t = t
#   model.states.fill(-1) # rest to ones; only interested in how mag is kept
#   # self.reset()
#   model.burnin(burninSamples)
#   res = np.asarray(model.simulate(nSamples))
#   H   = abs(res.mean())
#   HH =  ((res**2).mean() - res.mean()**2) * model.beta # susceptibility
#   return H, HH
#
#


# # deprecated
# def computeEntropyGivenMagnetization(samples, bins):
#     '''
#     Average of the nodes in the system states to get the
#     average magnetization, then make the distribution of the samples?
#     '''
#     avgSample  = samples.mean(axis = -1) # magnetization per sample
#     freq       = np.histogram(avgSample, bins = bins)[0] # normalizing produces odd results(?)
#     Z          = freq.sum()
#     nfreq      = freq / Z # normalize
#     return nfreq
#
# def magnetizationPerTemperature(graph, temperatures, nSamples = 1000, \
#                                 step = 20, bins = arange(0, 1, .1)):
#     '''
#     Returns the probability distribution of magnetization for
#     some samples
#     '''
#     magnetization = zeros( ( len(temperatures), len(bins) - 1) )
#     print(len(bins))
#     for tidx, temperature in enumerate(temperatures):
#         model = Ising(temperature = temperature, graph = graph)
#         samps = model.simulate(nSamples = nSamples, step = step)
#         print(samps.shape)
#         print(unique(samps, axis = 0))
#         magnetization[tidx, :] = computeEntropyGivenMagnetization(samples = samps, bins = bins)
#     return magnetization
# # deprecated
