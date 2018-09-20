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
import scipy
import networkx as nx, multiprocessing as mp, functools, tqdm, information
cimport cython
from models import Model
from libc.math cimport exp
from libc.stdlib cimport rand
cdef extern from "vfastexp.h":
    double exp_approx "EXP" (double)
cdef extern from "limits.h":
    int INT_MAX
# TODO: fix the equilibrium method
# TODO: further cythonization
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
        H  = np.zeros(self.nNodes, dtype = float)
        for node, nodeID in self.mapping.items():
            H[nodeID] = graph.nodes()[node]['H']

        # specific model parameters
        self.H              = H
        self.beta           = np.inf if temperature == 0 else 1/temperature
        self.betaThreshold  = betaThreshold
        self.verbose        = verbose
        self._t             = temperature
        self.states         = np.int16(self.states) # enforce low memory
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
    def energy(self, long node, np.ndarray states):
        '''
        input:
            :node: member of nodeIDs
        returns:
                :energy: current energy of systme config for node
                :flipEnergy: energy if node flips state
        '''
        neighboridx      = self.edgeData[node]
        interaction      = self.interaction[node]
        neighborStates   = states[neighboridx]
        H                = self.H[node] * states[node]
        # H                = 0
        # H                = self.H[neighboridx].dot(neighborStates)
        # computing (flip) energy per node
        # cdef float H = 0
        # loop declaration
        # compute hamiltonian
        # tmp = neighborStates.dot(interaction) / states[node]
        tmp = states[node] * neighborStates.dot(interaction)
        # tmp = neighborStates.sum()

        energy     = -tmp - H # empty list sums to zero
        flipEnergy = -energy

        # adding nudges
        nudge       = self.nudges[node] * states[node]
#        energy     -= (self.states[node] * nudge).sum()
        energy     -= nudge # TODO: checks to not allow for multiple nudges
        flipEnergy -= nudge
        return energy, flipEnergy
    # @jit
    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.nonecheck(False)
    def updateState(self, long [:] nodesToUpdate):
        '''
        Determines the flip probability
        p = 1/(1 + exp(-beta * delta energy))
        '''
        states        = self.states.copy() if self.mode == 'sync' else self.states # only copy if sync else alias
        cdef long  N = len(nodesToUpdate)
        cdef long  n
        cdef long node
        for n in range(N):
          node = nodesToUpdate[n]

          energy, flipEnergy  = self.energy(node, states)
          # TODO: change this mess
          # # heatbath glauber
          # if self.updateMethod == 'glauber':
          # tmp = self.beta * (flipEnergy - energy)
          tmp = - self.beta * energy * 2
          tmp = tmp if not np.isnan(tmp) else 0
          p = 1 / ( 1 + exp(tmp) )
          if rand() / float(INT_MAX) < p:
            states[node] = -states[node]

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
        if self.magSide == 'neg':
          self.states = states  if np.sign(states.mean()) < 0 else -states
        elif self.magSide == 'pos':
          self.states = -states  if np.sign(states.mean()) < 0 else states
        else:
          self.states = states
        return self.states # maybe aliasing again

    def computeProb(self):
        ''' Computes p_i = 1/z * (1 + exp( -beta * energy))**-1'''
        probs = np.zeros(self.nNodes)
        for node in self.nodeIDs:
            en = self.energy(node, self.states[node])[0]
            probs[node] = exp(-self.beta * en)
        return probs / np.nansum(probs)

    def matchMagnetization(self, targetMag =  1/np.e, \
                          temps = np.logspace(-3, 2, 20), n = int(1e3),
                          burninSamples = 100):

     '''
     Computes the magnetization for a range of temperature. This can be used to
     determine an optimal temperature with certain equilibrium properties
     Returns :temps: :mag: :sus:
     '''

     # start from 1s and check how likely nodes will flip
     H  = np.zeros( len(temps) )   # magnetization
     HH = np.zeros( ( len(temps ))) # susceptibility

     for idx, t in enumerate(tqdm.tqdm(temps)):
       self.t  = t # update temperature
       self.states = -np.ones(self.nNodes, dtype = self.states.dtype) # rest to ones; only interested in how mag is kept
       # self.reset()
       self.burnin(burninSamples)
       res = self.simulate(n)
       H[idx]  = abs(res.mean())                                # abs magnetization
       HH[idx] =  ((res**2).mean() - res.mean()**2) * self.beta # susceptibility
     return temps, H, HH

    def matchingEntropy(self, targetEntropy = 1/np.e, \
                       temps = np.logspace(-3, 1, 20), n = 10,\
                       ):
      def func(x, a, b, c, d):
        return   a / (1 + np.exp(b * (x - c)) ) + d
      def ifunc(x, a, b, c, d):
        return   c + 1/b * np.log(a / (x - d) - 1)
      H = np.zeros(len(temps))
      self.reset()
      energy = np.array([[self.energy(node, s)[0] for s in self.agentStates] for node in self.nodeIDs])
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
        self.nudges = np.zeros(self.nudges.shape)

def fitTemperature(temperature, graph, nSamples, step, fitModel = Ising):
    '''
    Used by match temperature to match the entropy
    for a specific graph of specific size
    '''
    model = fitModel(graph = graph, temperature = temperature, doBurnin = True)
    return information.getSnapShots(model = model, nSamples = nSamples,\
                                    step = step)[:2] # only keep the node probs

def matchTemperature(\
    graph,\
    temperatures = np.logspace(-20, 1, 10),\
    targetDistribution = None, \
    nSamples = 1000,\
    step = None,\
    fitModel = Ising):
    '''
    Estimates the temperature for which the average node probability [target]
    is achieved..
    The idea is to fit a model to the temperatures from which the target is idtsEstimated
    '''
    if type(step) is type(None):
        step = 5 * graph.number_of_nodes()
    probs = np.zeros((len(temperatures), graph.number_of_nodes(), 2))
    # for each T init model and burnin for equilibrium
    import multiprocessing as mp; import functools
    func = functools.partial(fitTemperature, graph = graph, nSamples = nSamples,\
                             step = step, fitModel = fitModel)
    with mp.Pool(processes = mp.cpu_count() - 1) as pool:
        probsr       = np.array(pool.map(func, temperatures), dtype = object)

    probs = np.array([p[1] for p in probsr])
    sprobs= np.array([np.array(list(p[0].values())) for p in probsr])
    H           = probs # information.entropy(probs).mean(-1) # average the entropies [temps  x nodes]
    Hstate = np.array([information.entropy(s) for s in sprobs]) # states have diff dimension

    func = lambda x, a, b, c, d: a + b - np.exp(-c * (x - d))
    from scipy.optimize import curve_fit
    from functools import partial
    try:
        coefs = curve_fit(func, temperatures, Hstate)[0]
        tmp = list(func.__code__.co_varnames) # dirty hacks!
        tmp.remove('x')
        ofunc =partial(func, **{i:j for i,j in zip(tmp, coefs)})
        half = np.log(2)/coefs[2] # not really half point, but approx
    except:
        print('not found')
        half = 0; ofunc = 0
    return Hstate, H, half, ofunc




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
