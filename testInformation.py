#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 19 09:27:38 2018

@author: casper
"""
import sys
from numpy import *
from matplotlib.pyplot import *
import unittest
import information
class Information(unittest.TestCase):
    '''
    Test module for different information theoretical measures
    '''
    def testEntropy(self):
        '''
        This tests the entropy of p = .5; should give 1
        '''
        p = array([.5, .5], ndmin = 2)
        h = information.entropy(p)
        self.assertEqual(h, 1)

    def testMutualInformation(self):
        '''
        Check that independent distributions yield 0 MI
        '''
        px = py = ones(2) * .5
        pxy = ones((2,2)) * .5
        I = information.mutualInformation(px, py, pxy)
        self.assertEqual(I, 0)

    def testParallelSnapShots(self):
        '''
        Tests whether the parallel chain produces the same results as the
        serial chain. The idea is to generate an iterable of nSamples long,
        and then divide this over 7 cpus; such that the cpu each performs one step.
        Do keep in mind that they have the same model, so each step mutates the states
        variable of the model. As such we can sample half of the normal samples because
        each step returns the zero step [mutated] and the first step.

        Using the Hellinger distance we can compare the two distributions against
        each other.

        Small notes, low samples and low step number prevents
        the checks from passing. Furthermore, not all states will be reached.
        However, if the step and samples are high enough the results will be the same
        with a huge speed boost on the sampling part.
        '''
        import fastIsing
        import networkx as nx
        # set up model
#        graph = nx.DiGraph()
        graph = nx.path_graph(5, create_using = nx.DiGraph())
#        attr        = dict(state = -1, H = 0, nudges = 0)
#        assignAttr  = {node : attr for node in graph.nodes()}
#        nx.set_node_attributes(graph, assignAttr)
        model = fastIsing.Ising(temperature = .21, graph = graph, doBurnin = True)

        # parameters for initial chain
        stdParams = dict(model = model,\
                        nSamples = 10000,\
                        step     = 1,\
                        useAllCores = True)
        # parallel attempt
        import time
        past = time.time()
        snapshotsParallel, nodeProbParallel, model = information.getSnapShots(**stdParams)
        print('Elapsed time parallel {}'.format(time.time() - past))

        # serial attempt
        stdParams['useAllCores'] = False
        stdParams['step']        = 10 * 2 ** model.nNodes
        stdParams['model']       = model

        past = time.time()
        snapshots, nodeProb, model = information.getSnapShots(**stdParams)
        print('Elapsed time serial {}'.format(time.time() - past))

        # hellinger distance calculation
        p1, p2 = zeros(len(snapshots)), zeros(len(snapshots))
        for idx, (key, value) in enumerate(snapshots.items()):
#            print(idx, key) # to know which is which
            p1[idx] = value; p2[idx] = snapshotsParallel.get(key, 0)
        from plotting import hellingerDistance
        h1     = hellingerDistance(p1, p2)
        h2     = hellingerDistance(nodeProbParallel, nodeProb)
        print('hellinger distance\nstate dist. {} \tnode dist. {}'.format(h1, h2))
        print('state prob serial \n\t {}\nstate prob parallel \n\t {}'.format(p1, p2))
        print('SSE state prob {} SSE node prob {}'.format(\
              sqrt((p1-p2)**2).sum(), sqrt((nodeProbParallel - nodeProb)**2).sum()))
        
        print('node prob serial \t\n {} \nnodeprob parallel \t\n {}'.format(nodeProb, nodeProbParallel)) # lazy formatting
        theta = 1;
        [self.assertLess(i, theta) for i in h2]
        self.assertLess(h1, theta)

if __name__ == '__main__':
    unittest.main()
   # import sys
   # sys.path.append('../')
   # print(sys.path)
   # from Model import fastIsing
   # # unittest.main()
   # test = InformationMeasures()
   # test.testParallelSnapShots()
   # show()
