from importPackages import *
import unittest
import plotting as plotz
import networkx as nx, fastIsing
'''
Checlist:
    - Do all the functions actually test or output something?
'''
class Ising(unittest.TestCase):
    def setUp(self):
        '''
        Generate a simple test graph
        '''
        import networkx as nx
        # graph = nx.path_graph(10)
        # for (i, j) in graph.edges():
            # graph[i][j]['J'] = 1
        graph = nx.DiGraph()
        graph = nx.path_graph(2, create_using = nx.DiGraph())
        # graph = nx.path_graph(2)
        # graph.add_edge(0, 1 , weight = 1)
        # graph.add_edge(0,0, J = 1)
        # graph.add_edge(1, 0, J = 1)
        # attr  = dict(state = -1, H = 0, nudges = 0)
        # attrN = dict(state = 1, H = 0, nudges = 2)
        # specify for specific nodes by another dict
        # consisting of nodeID : value dict
        # target = 10# nudge this node
        # assignAttr = {node : attr if node != target else attrN for node in graph.nodes()}
        # nx.set_node_attributes(graph, assignAttr)
        self.graph = graph
        self.model = fastIsing.Ising(graph, 1, False)
    def testSampling(self):
        import time
        s = time.time()
        self.model.simulate(10000, 1)
        print(time.time() - s)
    def testModelSetup(self):
        '''
        Use this to test model requirements, i.e. methods or variables
        that the model should have for information.py
        '''
        model = fastIsing.Ising(graph = self.graph, temperature = 1, doBurnin = False)
        # for key, value in model.__dict__.items(): print(key, value) # sanity
        # what do i expect from the model
        # TODO: change J to weight
        expectations = set('graph nudges edgeData states'.split(' '))
        for e in expectations:
            self.assertIn(e, model.__dict__)
    def testBurnin(self):
        model = fastIsing.Ising(graph = self.graph, temperature = 1, doBurnin = True)
    def testStateEnergyAtTemp(self):
        '''
        # shows: the entropy of state as a function of temperatures
               the entropy of the  nodes as a function of temperature
       '''
        # temperatures = logspace(start = -4, stop = 1, num = 20)
        # temperatures = arange(start = 0, stop = .5, step = .5)
        temperatures = logspace(-10, 1, 20)
#        temperatures = [0]
#        temperatures   = logspace(-6, 1, 20)
        # targetDist   = array([ .75, .25])
        targetDist   = array([.5, .5])
        root, half, H, sig = fastIsing.fitTemperatureToNodeProbability(\
                                                       targetDist, \
                                                       temperatures,\
                                                       graph = self.graph\
                                                        )

        x = linspace(0, max(temperatures))

        fig, ax = subplots();
        ax.plot(temperatures, H, '.')
        ax.plot(root, 0, '*r', markersize = 15)
        print(half)
        ax.plot(half, sig(half), '.r', markersize = 15)
        ax.plot(temperatures, sig(temperatures))
        setp(ax, **dict(title = 'root {}'.format(inf),  xlabel = 'temperature',\
                        ylabel = 'Entropy'))
        show()
        self.assertEqual(root, inf) # you expect entropy to be one at inf

    # def tearDown(self):
    #     plotz.saveAllFigures(useLabels = False, path = 'Figures/tests')

def closeTests():
     plotz.saveAllFigures(useLabels = False, path = '../Figures/tests')
if __name__ == '__main__':
   # unittest.main()
   a = Ising()
   a.setUp()
   a.testSampling()
