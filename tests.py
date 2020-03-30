import unittest as ut, networkx as nx
from PlexSim.Models.Models import Model
import subprocess, numpy as np
class TestBaseModel(ut.TestCase):
    def setUp(self):
        g = nx.path_graph(3)
        self.m = Model(graph = g)
        self.updateTypes = "single async sync".split()
    def sampling(self):
        """
        helper function
        """
        samples = self.m.sampleNodes(100)
        for sample in samples.base.flat:
            self.assertTrue(0 <= sample <= self.m.nNodes)
        return samples
    def test_init(self):
        for updateType in self.updateTypes:
            m = Model(graph = nx.path_graph(1), updateType = updateType)
    def test_updateTypes_sampling(self):
        """
        Test all the sampling methods
        It should generate idx from node ids
        """
        for updateType in self.updateTypes:
            self.m.updateType = updateType
            samples = self.sampling()
            # check the number of nodes being sampled
            if updateType == "single":
                self.assertEqual(samples.shape[-1],  1)
            else:
                self.assertEqual(samples.shape[-1], self.m.nNodes)
    def test_updateTypes_updateState(self):
        """
        Check whether the updateState function operates
        """
        for updateType in self.updateTypes:
            self.m.updateType = updateType
            update = self.m.updateState(self.m.sampleNodes(1)[0])
            # look for output
            self.assertTrue(update)
#@ut.skip("N")
from PlexSim.Models.FastIsing import Ising
class TestIsing(TestBaseModel):
    def setUp(self):
        g = nx.path_graph(1)
        g.add_edge(0,0)
        self.m = Ising(graph = g)
        #self.m = Potts(graph = g)
        self.updateTypes = "single async sync".split()
    def test_updateState(self):
        # force update to be independent
        # update all the nodes
        self.m.updateType = 'sync'
        temps = [0, np.inf]
        targs = [1, 0]
        for temp, target in zip(temps, targs) :
            # set temp
            self.m.t = temp
            # assign all states to be the same
            self.m.states = 1
            print(self.m.states.base, self.m.newstates.base)
            # store the before
            res = self.m.simulate(100000).mean()
            self.assertAlmostEqual(res, target, places = 1)
from PlexSim.Models.Potts import Potts
@ut.skip("Ising is doing weird things")
class TestPotts(TestIsing):
    def setUp(self):
        g = nx.path_graph(1)
        g.add_edge(0,0)
        self.m = Potts(graph = g)
        self.updateTypes = "single async sync".split()
    def test_updateState(self):
        # force update to be independent
        # update all the nodes
        self.m.updateType = 'sync'
        temps = [0, np.inf]
        targs = [1, 0.5]
        for temp, target in zip(temps, targs) :
            # set temp
            self.m.t = temp
            # assign all states to be the same
            self.m.states = 1
            # store the before
            res = self.m.simulate(10000).mean()
            self.assertAlmostEqual(res, target)

if __name__ == "__main__":
    ut.main()
