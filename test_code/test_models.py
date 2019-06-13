import unittest, sys, os
sys.path.insert(0, '../') # make paths available

from copy import deepcopy
class TestIsing(unittest.TestCase):
    """
    Sanity checking whether the deepcopy works
    """

    def setUp(self):
        from Models import fastIsing
        from Models import potts
        import networkx as nx

        g = nx.path_graph(5)

        self.modelprops = dict(\
                t = 2,\
                updateType = 'single',\
                nudgeType  = 'pulse',\
                magSide    = 'neg',\
                )

        self.model = fastIsing.Ising(graph = g, **self.modelprops)
    def test_key_properties(self):
        models = [deepcopy(self.model) for _ in range(2)]
        for name, prop in self.modelprops.items():
            for m in models:
                self.assertEqual(getattr(m, name), prop)

    def test_nudges(self):
        node = next(iter(self.model.graph.nodes()))
        test_nudges = {node : 1}
        self.model.nudges = test_nudges
        models = [deepcopy(self.model) for _ in range(5)]

        nodeidx = self.model.mapping[str(node)]
        for model in models:
            for node, idx in model.mapping.items():
                self.assertEqual(model.nudges[idx], test_nudges.get(node, 0))



class TestPotts(TestIsing):
    """
    Sanity checking whether the deepcopy works
    """

    def setUp(self):
        from Models import potts
        from Models import potts
        import networkx as nx

        g = nx.path_graph(5)

        self.modelprops = dict(\
                t          = 2,\
                updateType = 'single',\
                nudgeType  = 'pulse',\
                delta      = 2,\
                memorySize = 10,\
                )

        self.model = potts.Potts(graph = g, **self.modelprops)
        # print(dir(self.model))

if __name__ == '__main__':
    unittest.main()
