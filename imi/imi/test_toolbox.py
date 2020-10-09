import sys
sys.path.insert(0, '../')
from PlexSim.plexsim.models import *
import unittest as ut, networkx as nx

import numpy as np
from Toolbox import infcy
class Toolbox(ut.TestCase):
    model = Ising
    # g = nx.path_graph(5)
    g = nx.path_graph(3)
    def setUp(self):
        self.m = self.__class__.model(self.__class__.g)

    @ut.skip("skipping")
    def test_snapshots_and_entropy(self):
        # Case of infinite temperature
        N = 1000

        self.m.t = np.inf
        snapshots = infcy.getSnapShots(self.m, nSamples = N)

        target = 2 ** (self.m.nNodes - 1)
        px = np.fromiter(snapshots.values(), dtype = float)

        # test entropy too
        H = infcy.entropy(px)
        self.assertAlmostEqual(H, target, 1)

        # Case of zero temperature
        self.m.t = 0
        self.m.states = self.m.agentStates[0]
        snapshots = infcy.getSnapShots(self.m, nSamples = N)
        px = np.fromiter(snapshots.values(), dtype = float)

        H = infcy.entropy(px)
        self.assertAlmostEqual(H, 0, 1)

    def test_case_copy_bit(self):
        # set states at specific
        start = np.ones(self.m.nNodes)
        # we would expect 1 to flip with probability
        # 1 / (1 + exp(-2beta))
        nTrials = 10000
        results = np.zeros((self.m.nNodes, self.m.nStates)) 

        # setup model
        # let 1 copy the start of 0
        self.m.t = 0
        self.m.updateType = 'sync'
        
        mapper = {i : idx for idx, i in enumerate(self.m.agentStates)}
        # only update B
        CHECK = memoryview(np.array([1]))
        for trial in range(nTrials):
            # reset state
            self.m.states = start
            tmp = self.m.updateState(CHECK)
            for node, state in enumerate(tmp):
                results[node, mapper[state]] += 1 / nTrials

        desiredDists = [[1., 0], [.5, .5]]
        print('\n'*2)
        for node, r in enumerate(results):
            for observed, desired in zip(r, desiredDists[node]):
                self.assertAlmostEqual(observed, desired, 1)

    
    def test_reverse(self):
        kwargs = dict(\
                      nSteps = 1000,\
                      window = 10)
        output = infcy.reverseMC(self.m, **kwargs)

    def test_doTrial(self):
        print("testing doTrial")
        kwargs = dict(\
                      nTrials = 1,\
                      nSamples = 10,\
                      nudgeSizes = np.array([0.]),\
                      reverse = True,\
                      center = True,\
                      )
        output = infcy.doTrial(self.m, **kwargs)
        self.assertTrue(isinstance(output, dict))
if __name__ == "__main__" :
    ut.main()
