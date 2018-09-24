#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 19 09:28:31 2018

@author: casper
"""
import unittest
from numpy import *
from matplotlib.pyplot import *
import networkx as nx
import information
import plotting as plotz
from XOR import *
from fastIsing import Ising
from multiprocess import cpu_count
class SimpleGraph(unittest.TestCase):
    '''
    Test class for a simple graph
    '''
    def setUp(self, showMe = True, mode = 'async', T =  1, model = Ising):
        '''
        Create the simple graph run some simulations
        0 -----------> 1
            weight=1
        H_i = 0 for all i in N
        All states start in 1,1
        '''
        import networkx as nx
        # TODO: tmp hack
        if model is Ising:
    #        graph = nx.DiGraph()
    #        [graph.add_edge(0, i) for i in range(1, 6)]
            graph= nx.DiGraph()
            s, ss = 15, .1
#            graph = nx.powerlaw_cluster_graph(5, 4, .1)
            graph = nx.path_graph(3 , nx.DiGraph())
            # graph = nx.path_graph(2)
#            graph = nx.path_graph(5, nx.DiGraph())
#            graph = nx.path_graph(3, nx.DiGraph())
#            graph.add_edge(0, 1, weight = s)
#            graph.add_edge(0, 2, weight = s)
#            graph.add_edge(1, 2, weight = s)
#
##            graph.add_edge(2, 0, weight = ss)
#            graph.add_edge(0, 0, weight = ss)
#            graph.add_edge(1, 1, weight = ss)_
#            graph.add_edge(2, 2, weight = ss)

#            graph.add_edge(0, 4)
#            graph = nx.DiGraph(); graph.add_edge(0, 1, weight = 1); graph.add_edge(2, 1, weight = -1) # XOR case
            if T == None:
                temperatures = logspace(-3, 2, 10)
                _, T, H, optsig =  fastIsing.matchTemperature(graph, temperatures)
            self.model = model(graph, temperature = T, doBurnin = True, mode = mode)
        else:
            self.model = model(mode = mode)
        self.showMe = showMe

        sig = lambda x, a, b, c: a / (1 + exp(b * x - c))
        self.func = lambda x, a, b, c, d, e, f: a * exp(b * (x - c)) + d * exp(e * (x - f))
        self.func = lambda x, a, b, c, d, e, f: a * exp(b *x**c)

        # standard parameters
        self.deltas   = 10
        self.repeats = 100
        self.nSamples = 1_00000
        self.step     = 1

        self.parallel = cpu_count()
        self.snapshots = information.getSnapShots(model = self.model, nSamples = self.nSamples, step = self.step, parallel = self.parallel)[0]
        print(len(self.snapshots))
        self.mc        = information.monteCarlo(\
        model = self.model, snapshots = self.snapshots, \
        deltas = self.deltas, repeats = self.repeats, pulse = {},\
        parallel = self.parallel)
        return self.model

    def testMutualInformationSinc(self):
        '''
        tests  I(x_i^t-delta ; X^t )
        '''
        pNode, pState, pxgiveny, joint, I = information.mutualInformationShift(model = self.model,\
        snapshots = self.snapshots, montecarloResults = self.mc, mode = 'sinc')
        # ax.set_yscale('log')
        idts = plotz.showFit(self.model, I.T, func)[0]
        if self.showMe:
            show()
        self.assertLess(idts[0,0], idts[1,0])
        return I, idts, pNode, pState, pxgiveny

    def testMutualInformationSource(self):
        '''
        tests  I(x_i^t ; X^-delta  )
        # TODO: clean up
        '''
        pNode, pState, pxgiveny, joint, I = information.mutualInformationShift(model = self.model,\
        snapshots = self.snapshots, montecarloResults = self.mc, mode = 'source')
        sig = lambda x, a, b: 1 / ( 1 + exp(x - b))
        func = self.func
       # func = lambda x, a, b  : 1 / (1 + exp(-a * x - b))
        # func = lambda x, a, b, c: a * exp(-b * x - c)
#        print(I.shape)
#        print(coefs)
#        ax.set_xlim([-1, self.stdParams['deltas']]); ax.set_ylim([0, 1])
        # ax.set_yscale('log')
        [self.assertAlmostEqual(all(i.sum(-1)), 1) for i in [pNode, pState, pxgiveny]]
        idts = plotz.showFit(self.model, I.T, func)[0]
        if self.showMe:
            show()
        self.assertLess(idts[1, 0], idts[0, 0])
        return  I, idts, pNode, pState, pxgiveny


    def testPulse(self):
        '''
        Test pulse function, here only 1 time step contains a nudge
        '''
        pulseSize = 1
        # pulse every node
        pulses    = {node : { node : pulseSize} for node in self.model.mapping.keys()}
        pulses['control'] = {} # add a control condition [no pulse]

        import itertools
        targets = list(itertools.combinations(self.model.mapping.keys(), 2))
        targets.append((0, 1, 2))
        for target in targets:
            pulses[target] = {t : pulseSize for t in target}

#        pulses = {(0, 1) : {(0, 1) : pulseSize}, 0 : {0 : pulseSize}}
        results = {} # keep the formatting in line with some of the test functions
        for label, pulse in pulses.items():
            pulseResult    = information.nudgeOnNode(\
                         pulse, **self.stdParams, mode = 'state', pulse = True)
            results[label] = pulseResult
            self.model.reset(doBurnin = True, useAtleastNSamples = 100)
        # test whether the mutual information of the pulse is less, than when the nodes
        # is pulsed; last indeAx is mutual information
        return results
    def testNudge(self):
        '''
        Test nudge function, a nudge here is defined as an external intervention that lasts
        for the entire duration of the montecarlo samples
        '''
        nudgeSize = 1
        # nudge every node
        nudges    = {node : { node : nudgeSize} for node in self.model.mapping.keys()}
        nudges['control'] = {} # add a control condition [no pulse]

        import itertools
        targets = list(itertools.combinations(self.model.mapping.keys(), 2))
        targets.append((0, 1, 2))
        for target in targets:
            nudges[target] = {t : nudgeSize for t in target}
        results = {} # keep the formatting in line with some of the test functions
        for label, nudge in nudges.items():
            nudgeResult    = information.nudgeOnNode(\
                         nudge, **self.stdParams, mode = 'state', pulse = False)
            results[label] = nudgeResult
            self.model.reset(doBurnin = True, useAtleastNSamples = 100)

        return results
    # TODO: remove this function
    # def testPulseVsNudge(self):
    #     results = {}
    #     targets = list(self.model.mapping.keys())[0] # define test cases
    #     effect  = inf
    #     info = {0 : effect}
    #     # ugly
    #     results['control'] = information.nudgeOnNode({}, **self.stdParams,\
    #                                                  mode = 'state', pulse = False)
    #
    #     results['nudge'] = information.nudgeOnNode(info, **self.stdParams,\
    #                                                 mode = 'state', pulse = False)
    #     results['pulse'] = information.nudgeOnNode(info, **self.stdParams,\
    #                                                 mode = 'state', pulse = True)
    #     return results

    def testCondition(self):
        conditions = None # remember that this must be done according to the nodes in the graph however right now they overlap
        joint, I = information.mutualInformationShift(self.model, self.snapshots, self.mc, conditions = conditions, parallel = False)
        res = {}
        res['control'] = dict(I = I, mc = self.mc)
        return res



if __name__ == '__main__':
   # unittest.main()
    import fastIsing
    import functools
    test = SimpleGraph()
    test.setUp()
    # test.model.updateState = functools.partial(test.model.updateState, nodesToUpdate = [0, 2])
    # res = test.testNudge()
    # res = test.testPulseVsNudge()
    import time
    past = time.time()
    res = test.testCondition()
    print(res['control']['I'].shape)
    f = lambda x, a, b, c, d, e, f: a * exp(-b * (x - c))  #+ d * exp(-e * (x - f))
    plotz.showFit(test.model, res['control']['I'].T, f)
    show()
#    print(time.time() - past)
#    print('done')
#    p = res['control']['node']
#    print(p)
#    t = plotz.hellingerDistance(p[..., [0]], p[...,[1]])
#    pp = res['control']['pxy']
#    #%%
##    close('all')
#    tt = plotz.hellingerDistance(pp[..., [0]], pp[..., [1]])
#    ttt = nanmean(tt, -1)
#
#    fig, ax = subplots()
#    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
#    axx = inset_axes(ax,  width = 4.4, height = 2.4, loc = 'upper right')
#
#    colors = cm.get_cmap('tab20')(arange(pp.shape[1]))
#    [ax.plot(i, '.', markersize = 25, color = c) for i, c in zip(ttt.T, colors)]
#
#    plotz.addGraphPretty(test.model, axx)
##    pos = nx.circular_layout(test.model.graph)
##
##    nx.draw_networkx_nodes(test.model.graph, pos,  node_color = colors, label = True)
##    nx.draw_networkx_edges(test.model.graph, pos)
#    setp(ax, xlabel = 'time [step]', ylabel = '$<H(P,Q)>_{X}$')
#    axx.set_xlabel('Graph')
#    axx.axis('off')
##    axx.set_xticks([]); axx.set_yticks([])
#
#    print(pp.shape)
#    # %%
##    close('all')
#    func = lambda x, a, b, c, d, e, f, g, h, i :\
#    a * exp(b * x **c)
##    a * exp(b * (x - c)) + d * exp(e * (x - f)) + g * exp(h * (x - i))
#    # func = lambda x, a, b, c: a * exp(-b * (x - c))
#    styles = '- -- -.'.split(' ')
#    colors = cm.get_cmap('tab20')(arange(test.model.nNodes))
#    x = linspace(0, 10, 1000)
#    fig, ax = subplots();
#    for style, (key, value) in zip(styles, res.items()):
#        idts, coefs = plotz.fit(value['I'], func)
#        [ax.plot(x, func(x, *coef), color = color, linestyle = style, label = key) for coef, color in zip(coefs, colors)]
#        [ax.plot(i, '.', color = color, markersize = 20) for i, color in zip(value['I'].T, colors)]
#    ax.legend()
##    ax.set_xlim(0, 5)
#
#    show()
