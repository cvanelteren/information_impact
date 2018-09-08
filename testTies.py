# from ties import computeStrengthOfTies
# from numpy import mean
# import networkx as nx
# import unittest
#
# class testTieStrength(unittest.TestCase):
#     def setUp(self):
#         self.n = 10
#         self.meanStrength = lambda G: mean(computeStrengthOfTies(G))
#     def testFullConnected(self):
#         ''' Test fully connected graph.'''
#         G = nx.random_graphs.complete_graph(self.n)
#         print(self.meanStrength(G))
#         assert self.meanStrength(G) == 1
#     def testLineGraph(self):
#         '''Test line graph.'''
#         G = nx.path_graph(self.n)
#         assert  self.meanStrength(G) == 0
