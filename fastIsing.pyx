#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  6 09:36:17 2018

@author: Casper van Elteren
"""
# cython: infer_types=True
import numpy  as np
cimport numpy as np

from scipy.stats import linregress
import networkx as nx, multiprocessing as mp, functools, tqdm, information, scipy,  functools, copy
cimport cython
from cython import parallel
from models import Model
from libc.math cimport exp
from libc.stdlib cimport rand
cdef extern from "vfastexp.h":
    double exp_approx "EXP" (double)
cdef extern from "limits.h":
    int INT_MAX
# TODO: this is still too pythonic. The conversion was made from direct python code; future me needs to hack this into c
class Ising(Model):
    def __init__(self, graph, temperature, doBurnin = False, \
                 betaThreshold = 1e-2, agentStates = [-1 ,1],
                 mode = 'async', verbose = False, \
                 nudgeMode = 'constant', magSide = 'neg'):

        super(Ising, self).__init__(\
                                  graph = graph, \
                                  agentStates = agentStates, \
                                  mode = mode, verbose = verbose, \
                                  nudgeMode = nudgeMode)

        hPresent = True if nx.get_node_attributes(graph, 'H') != {} else False
        # TODO: redundant property; only present for matplotlib

        if not hPresent:
            if verbose : print('external field not detected assuming 0')
            nx.set_node_attributes(graph, 0, 'H')
        H  = np.zeros(self.nNodes)
        for node, nodeID in self.mapping.items():
            H[nodeID] = graph.nodes()[node]['H']

        # specific model parameters
        self.H              = H
        self.beta           = np.inf if temperature == 0 else 1/temperature
        self.betaThreshold  = betaThreshold
        self.verbose        = verbose
        self._t             = temperature
        self.statesDtype    = np.int64
        self.states         = self.statesDtype(self.states) # enforce low memory
        self.magSide        = magSide

        # define the update method
        # self.updateMethods  = dict(glauber = self.energy_default, \
                              # glauber_neighbor = self.energy_neighbor)

        # self.energy         = self.updateMethods[updateMethod]
        if verbose :
          print(\
              f'Initialized ISING model with \ntemperature: {temperature:>0}\nmode: {mode:>0}')
        if doBurnin:
          self.burnin()
        else:
            print('No burnin used')
    # added property cuz im lazy
    @property
    def t(self):
        return self._t
    @t.setter
    def t(self, value):
        self._t = value
        self.beta = 1/value if value != 0 else np.inf
        if self.verbose :
          print(f'Setting T = {value}, beta = {self.beta}')

    def burnin(self, useAtleastNSamples = None):
        '''
        Go to equilibrium distribution; uses magnetization and linear regression
        to see if the system is stabel
        '''
        if useAtleastNSamples is None:
            useAtleastNSamples = int(1e4)
        if self.verbose:
          print(f'Equilibrating using N = {useAtleastNSamples}')
        # evolve the state until the magnetization has stabilized
        counter = 0

        # magnetization function
        magnetization = lambda x: np.mean(x) # count up states

        y  = np.array(magnetization(self.states))

        if self.verbose: print('Starting burnin')
        alpha = True
        while alpha:
            states = self.updateState(self.sampleNodes[self.mode](self.nodeIDs))
            y      = np.hstack((y, magnetization(states)))
            if counter > useAtleastNSamples : # run atleast 10 samples
                # do linear regression
                h = len(y + 2) # plus 2 due to starting point
                x = np.arange(h)
                beta = linregress(x, y).slope
                if abs(beta) < self.betaThreshold:
                    alpha = False
            counter += 1
        else:
            if self.verbose :
                print('Number of bunin samples used {0}\n\n'.format(counter))
                print(f'absolute mean magnetization last sample {abs(y[-1])}')
            if counter == 0:
                if self.verbose: print('WARNING: no burnin used')
        return y

    def reset(self, useAtleastNSamples = None):
        '''Resets the states to random'''
        self.states = np.random.choice(self.agentStates, size = self.nNodes)
        if useAtleastNSamples is not None:
            self.burnin(useAtleastNSamples)


    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.nonecheck(False)
    def energy(self, int node, long [:] states):
        '''
        input:
            :node: member of nodeIDs
        returns:
                :energy: current energy of systme config for node
                :flipEnergy: energy if node flips state
        '''
        return c_energy(node, states, self.edgeData[node], self.interaction[node], self.H, self.nudges[node])
        # cdef int [   :] neighboridx    = self.edgeData[node]
        # cdef double [:] interaction    = self.interaction[node]
        # cdef long [   :] neighborStates = states[neighboridx]
        # cdef double [:] H              = self.H[neighboridx].dot(neighborStates)
        # # computing (flip) energy per node
        # cdef double tmp   = states[node] * neighborStates.dot(interaction)
        #
        # energy     = -tmp - H # empty list sums to zero
        #
        # # adding nudges
        # nudge       = self.nudges[node] * states[node]
        #
        # energy     = energy - nudge # TODO: checks to not allow for multiple nudges
        # return energy
    # @jit
    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.nonecheck(False)
    def updateState(self, int [:] nodesToUpdate):
        '''
        Determines the flip probability
        p = 1/(1 + exp(-beta * delta energy))
        '''


        cdef long [:] states           = self.states.copy() if self.mode == 'sync' else self.states # only copy if sync else alias
        cdef long [:] newstates        = self.states.copy() if self.mode == 'sync' else self.states # only copy if sync else alias
        cdef long  N = len(nodesToUpdate)
        cdef long  n
        cdef long node
        cdef double energy, p
        # not sure if this helps
        cdef calcE       = self.energy
        cdef str magSide = self.magSide
        cdef double beta = self.beta
        cdef double nudge

        cdef edgeData    = self.edgeData
        cdef interaction = self.interaction
        cdef  H          = self.H

        for n in range(N):
          node  = nodesToUpdate[n]
          nudge = self.nudges[n]
          energy = c_energy(node, states, edgeData[node],\
                            interaction[node], H, nudge)
          # TODO: change this mess
          # # heatbath glauber
          # if self.updateMethod == 'glauber':
          # tmp = - self.beta * energy * 2
          if np.isnan(energy):
              p = 0.5
          else:
              p = 1 / ( 1 + exp(-beta *  2 * energy) )
          # if np.random.rand() < p:
          if rand() / float(INT_MAX) < p: # faster
            newstates[node] = -states[node]

          # elif self.updateMethod == 'metropolis':
            # MCMC moves
          # cost = (flipEnergy - energy)
          # if cost <= 0:
          #   states[node] = -states[node]
          # else:
          #   betaDeltaEnergy     = self.beta * (flipEnergy - energy)
          #   betaDeltaEnergy     = 0 if np.isnan(betaDeltaEnergy) else betaDeltaEnergy
          #   # Gibbs-sampling; heathbath equation
          #   p = exp(-self.beta * (flipEnergy - energy))
          #   if rand() / float(INT_MAX)  <= p:
          #     states[node] = -states[node]
        # TODO: ugly
        mu = np.mean(newstates)

        cdef int idx = 1
        if mu < 0 and magSide == 'pos':
            idx = -1
        elif mu > 0 and magSide == 'neg':
            idx = -1
        self.states = np.multiply(newstates, idx)
        return self.states

    def computeProb(self):
        """
        Compute the node probability for the current state p_i = 1/z * (1 + exp( -beta * energy))**-1
        """

        probs = np.zeros(self.nNodes)
        for node in self.nodeIDs:
            en = self.energy(node, self.states[node])
            probs[node] = exp(-self.beta * en)
        return probs / np.nansum(probs)

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
     H, HH = results.T

     return H, HH

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



    def matchingEntropy(self, targetEntropy = 1/np.e, \
                       temps = np.logspace(-3, 1, 20), n = 10,\
                       ):
      """
      Matches the temperature to :targetEntropy:. This function is deprecated
      """
      def func(x, a, b, c, d):
        return   a / (1 + np.exp(b * (x - c)) ) + d
      def ifunc(x, a, b, c, d):
        return   c + 1/b * np.log(a / (x - d) - 1)
      H = np.zeros(len(temps))
      self.reset()
      energy = np.array([[self.energy(node, s) for s in self.agentStates] for node in self.nodeIDs])
      for tidx, t in enumerate(temps):
        self.t = t
        probs   = 1 / (1 + np.exp(self.beta * energy))
        H[tidx] = np.mean(-np.nansum(probs * np.log2(probs), axis = 1))
      try:
          a, b = scipy.optimize.curve_fit(func, temps, H,  maxfev = 10000)
          matchedEntropy = ifunc(targetEntropy, *a)
          return temps, H, matchedEntropy
      except Exception as e:
            print(e)
            return temps, H


    def removeAllNudges(self):
        '''
        Sets all nudges to zero
        '''
        self.nudges = np.zeros(self.nudges.shape)
# @cython.boundscheck(False)
# cdef np.ndarray c_updateState(int [:] nodesToUpdate,\
# int [:] states,\
# double [:] interactions,\
# int [:] edgeData,\
# double beta,\
# double [:] H,\
# str magSide,\
# str mode ):
#     '''
#     Determines the flip probability
#     p = 1/(1 + exp(-beta * delta energy))
#     '''
#
#     states = states.copy() if mode == 'sync' else states # only copy if sync else alias
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
@cython.boundscheck(False)
cdef double c_energy(int node, long[:] states,\
                      int [:] edgeData,\
                      double [:] interaction,\
                      double [:] H, double nudge):

  cdef double energy = 0
  cdef long N = len(edgeData)
  cdef double _inter, _H
  cdef long _edge, _state
  cdef int i
  cdef long _nodeState = states[node]
  for i in parallel.prange(N, nogil = True):
    _inter = interaction[i]
    _edge  = edgeData[i]
    _state = states[i]
    _H    = H[_edge]
    energy -= _nodeState * _state * _inter * _edge + _H * _state
  energy -= nudge
  return energy



def fitTemperature(temperature, graph, nSamples, step, fitModel = Ising):
    '''
    Used by match temperature to match the entropy
    for a specific graph of specific size
    '''
    model = fitModel(graph = graph, temperature = temperature, doBurnin = True)
    return information.getSnapShots(model = model, nSamples = nSamples,\
                                    step = step)[:2] # only keep the node probs


def matchMagnetization(modelt, nSamples, burninSamples):
  '''
  compute in parallel the magnetization
  '''
  model, t = modelt
  model.t = t
  model.states.fill(-1) # rest to ones; only interested in how mag is kept
  # self.reset()
  model.burnin(burninSamples)
  res = np.asarray(model.simulate(nSamples))
  H   = abs(res.mean())
  HH =  ((res**2).mean() - res.mean()**2) * model.beta # susceptibility
  return H, HH




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
