#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  6 09:36:17 2018

@author: casper
"""
__author__ = 'Casper van Elteren'
from numpy import *
import networkx as nx, multiprocess as mp, functools, tqdm, information
from models import Model
from numba import jit
class Ising(Model):
    def __init__(self, graph, temperature, doBurnin = True, \
                 betaThreshold = 1e-2, agentStates = [-1,1],
                 mode = 'async', verbose = False):
        super(Ising, self).__init__(graph = graph, \
             agentStates = agentStates, mode = mode, verbose = verbose)
        hPresent = True if nx.get_node_attributes(graph, 'H') != {} else False
        # TODO: redundant property; only present for matplotlib

        if not hPresent:
            if verbose : print('external field not detected assuming 0')
            nx.set_node_attributes(graph, 0, 'H')
        H  = zeros(self.nNodes, dtype = float)
        for node, nodeID in self.mapping.items():
            H[nodeID] = graph.nodes()[node]['H']

        # specific model parameters
        self.H              = H
        self.beta           = inf if temperature == 0 else 1/temperature
        self.betaThreshold  = betaThreshold
        self.verbose        = verbose
        self._t             = temperature
        if verbose :
            print(\
          'Initialized ISING model with \ntemperature: {:>0}\nmode: {:>0}'\
          .format(temperature, mode))

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
        self.beta = 1/value if value != 0 else inf
        print(f'Setting T = {value}, beta = {self.beta}')
    def burnin(self, useAtleastNSamples = None):
        '''
        Go to equilibrium distribution; uses magnetization and linear regression
        to see if the system is stabel
        '''
        if useAtleastNSamples is None:
            useAtleastNSamples = 2 ** self.nNodes
        # evolve the state until the magnetization has stabilized
        counter = 0
        magnetization = lambda x: sum(x) # count up states
        y  = array(magnetization(self.states))
        if self.verbose: print('Starting burnin')
        alpha = True
        while alpha:
            states = self.updateState()
            y      = hstack((y, magnetization(states)))
            if counter > useAtleastNSamples : # run atleast 10 samples
                # do linear regression
                h = len(y + 2)
                x    = hstack( ( arange(h)[:, None], ones((h, 1)) ) )
                beta = linalg.pinv(x.T.dot(x)).dot(x.T).dot((y - cumsum(y))/len(y - 1))  # WARNING PINV!!!
                if abs(beta[0]) < self.betaThreshold:
                    alpha = False
            counter += 1
        else:
            if self.verbose :
                print('Number of bunin samples used {0}\n\n'.format(counter))
            if counter == 0:
                if self.verbose: print('WARNING: no burnin used')
        return y

    def reset(self, doBurnin = True, useAtleastNSamples = None):
        '''Resets the states to random'''
        self.states = random.randint(0, 2, size = self.nNodes) * 2 - 1
        if doBurnin:
            self.burnin(useAtleastNSamples)
    @jit
    def energy(self, node):
        '''
        input:
            :node: member of nodeIDs
        returns:
                :energy: current energy of systme config for node
                :flipEnergy: energy if node flips state
        '''
        neighoridx          = self.edgeData[node]
        interaction         = self.interaction[node]
        neighborStates      = self.states[neighoridx]
        Hij                 = self.H[neighoridx] # stupid integer bullshite

        # computing (flip) energy per node
        H          = neighborStates.dot(Hij) # external magnetic field
        tmp        = self.states[node] * neighborStates.dot(interaction)
        energy     = -tmp - H # empty list sums to zero
        flipEnergy = tmp - H

        # adding nudges
        nudge       = self.nudges[node]
#        energy     -= (self.states[node] * nudge).sum()
        energy     -= nudge # TODO: checks to not allow for multiple nudges
        flipEnergy -= nudge
        return energy, flipEnergy
    @jit
    def updateState(self, nodesToUpdate = None):
        '''
        Determines the flip probability
        p = 1/(1 + exp(-beta * delta energy))
        '''
        # TODO: leaave open the possibility to update one node
#        if type(nodesToUpdate) is type(None):
        if type(nodesToUpdate) is type(None):
            nodesToUpdate = self.sampleNodes[self.mode](self.nodeIDs) # use mode when the model was defines
        states        = self.states.copy() if self.mode == 'sync' else self.states # only copy if sync else alias
#        print(nodesToUpdate)
        for node in nodesToUpdate:
            energy, flipEnergy  = self.energy(node)
            betaDeltaEnergy     = -self.beta*(flipEnergy - energy)
            betaDeltaEnergy     = 0 if isnan(betaDeltaEnergy) else betaDeltaEnergy
            p                   = float_power(1 + exp(betaDeltaEnergy), -1) # temp -> 0 beta -> inf exp (-beta) -> 0
            # if the new energy energy is more only keep with prob p
#            p = 1/2 if isnan(p) else p # inf * 0 = nan, should be 1/2 using l'hopital (exp(0)))
            self.states[node] = states[node] if random.rand() <= p  else -states[node]
        return self.states # maybe aliasing again

    def computeProb(self):
        ''' Computes p_i = 1/z * (1 + exp( -beta * energy))**-1'''
        probs = zeros(self.nNodes)
        for node in self.mapping.values():
            en = self.energy(node)[0]
            probs[node] = exp(-self.beta * en)
        return probs / nansum(probs)

    def removeAllNudges(self):
        self.nudges = zeros(self.nudges.shape)

def fitTemperature(temperature, graph, nSamples, step, fitModel = Ising):
    '''
    Used by match temperature to match the entropy
    for a specific graph of specific size
    '''
    model = fitModel(graph = graph, temperature = temperature, doBurnin = True)
    return information.getSnapShots(model = model, nSamples = nSamples,\
                                    step = step)[:2] # only keep the node probs

def matchTemperature(\
    graph: 'networkx graph',\
    temperatures: 'range over which to simulate' = logspace(-20, 1, 10),\
    targetDistribution: 'matched to entropy of this distribution' = None, \
    nSamples: 'number of samples'       = 1000,\
    step: 'step size between samples'    = None,\
    fitModel = Ising):
    '''
    Estimates the temperature for which the average node probability [target]
    is achieved..
    The idea is to fit a model to the temperatures from which the target is idtsEstimated
    '''
    if type(step) is type(None):
        step = 5 * graph.number_of_nodes()
    probs = zeros((len(temperatures), graph.number_of_nodes(), 2))
    # for each T init model and burnin for equilibrium
    import multiprocessing as mp; import functools
    func = functools.partial(fitTemperature, graph = graph, nSamples = nSamples,\
                             step = step, fitModel = fitModel)
    with mp.Pool(processes = mp.cpu_count() - 1) as pool:
        probsr       = array(pool.map(func, temperatures), dtype = object)

    probs = array([p[1] for p in probsr])
    sprobs= array([array(list(p[0].values())) for p in probsr])
    H           = probs # information.entropy(probs).mean(-1) # average the entropies [temps  x nodes]
    Hstate = array([information.entropy(s) for s in sprobs]) # states have diff dimension

    func = lambda x, a, b, c, d: a + b - exp(-c * (x - d))
    from scipy.optimize import curve_fit
    from functools import partial
    try:
        coefs = curve_fit(func, temperatures, Hstate)[0]
        tmp = list(func.__code__.co_varnames) # dirty hacks!
        tmp.remove('x')
        ofunc =partial(func, **{i:j for i,j in zip(tmp, coefs)})
        half = log(2)/coefs[2] # not really half point, but approx
    except:
        print('not found')
        half = 0; ofunc = 0
    return Hstate, H, half, ofunc




# deprecated
def computeEntropyGivenMagnetization(samples, bins):
    '''
    Average of the nodes in the system states to get the
    average magnetization, then make the distribution of the samples?
    '''
    avgSample  = samples.mean(axis = -1) # magnetization per sample
    freq       = histogram(avgSample, bins = bins)[0] # normalizing produces odd results(?)
    Z          = freq.sum()
    nfreq      = freq / Z # normalize
    return nfreq

def magnetizationPerTemperature(graph, temperatures, nSamples = 1000, \
                                step = 20, bins = arange(0, 1, .1)):
    '''
    Returns the probability distribution of magnetization for
    some samples
    '''
    magnetization = zeros( ( len(temperatures), len(bins) - 1) )
    print(len(bins))
    for tidx, temperature in enumerate(temperatures):
        model = Ising(temperature = temperature, graph = graph)
        samps = model.simulate(nSamples = nSamples, step = step)
        print(samps.shape)
        print(unique(samps, axis = 0))
        magnetization[tidx, :] = computeEntropyGivenMagnetization(samples = samps, bins = bins)
    return magnetization
# deprecated
