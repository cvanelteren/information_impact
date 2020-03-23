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
from PlexSim.Models.FastIsing import Ising
class TestIsing(TestBaseModel):
    def setUp(self):
        g = nx.path_graph(3)
        self.m = Ising(graph = g)
        self.updateTypes = "single async sync".split()
    def test_updateState(self):
        # force update to be independent
        # update all the nodes
        self.m.updateType = 'sync'
        for temp in [0, np.inf]:
            # set temp
            self.m.t = temp
            # assign all states to be the same
            self.m.states = 1
            # store the before
            before = self.m.states.base.copy()
            # update
            after  = self.m.updateState(self.m.sampleNodes(1)[0])
            after  = self.m.updateState(self.m.sampleNodes(1)[0])
            # logics
            if temp == 0:
                self.assertTrue(all(before == after), f"{temp}")
            else:
                self.assertTrue(all(before != after), f"{temp}\nbefore:{before}\nAfter: {after.base}")
            print(after.base, before)
if __name__ == "__main__":
    ut.main()
