#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 19 09:26:51 2018

@author: casper
"""
from numpy import *
from matplotlib.pyplot import *
import plotting as plotz
import networkx as nx,  unittest
class Plotting(unittest.TestCase):
    '''
    Test class for the plotting toolbox
    '''
    def setUp(self):
        func = lambda x, a, b, c: a + b * exp(- c * x)
        x    = linspace(0, 10, 1000)
        y    = func(x, 0, 1, 1)[:, None]# add some noise
#        print(y.shape); assert 0
        self.trueY = y
        self.empY  = y + random.randn(len(x))[:, None] * 1e-2
        self.func  = func
        self.x     = x
        graph = nx.DiGraph()
        graph.add_edge(0, 1 , weight = 1)
        for node in graph.nodes():
            graph.nodes()[node]['state'] = 1 if random.rand() < .5 else -1
        self.graph = graph
    def testShowGraph(self):
        plotz.showGraph(self.graph)

    def testIDT(self):
        '''
        Tests the fit exponential function based on some test data
        Do multiple runs to check stability
        TODO CHECK THIS
        '''
        stabilityAsymptote = []
        fig, ax = subplots();
        for i in range(10):
            idtsEstimated, popt = plotz.fit(self.empY, self.func, x = self.x)
            popt = tuple(popt[0])

            recov         = self.func(self.x, *popt) # estimated based on the noisy data

            xAsymptote    = idtsEstimated[0,0] #        print(xAsymptote)
            ySymp         = self.func(xAsymptote, *popt)
            xHalf         = idtsEstimated[0,1]
            yHalf         = self.func(xHalf, *popt)
            cheapTote     = where(abs(diff(recov)) < 1e-5)[0][0]
            xCheapTote    = self.x[cheapTote]
            yCheapTote    = self.func(xCheapTote, *popt)
            # print(xAsymptote, ySymp, xCheapTote, yCheapTote)
            stabilityAsymptote.append(xAsymptote)
            # # show the functions
            ax.plot(self.x, self.trueY, self.x, self.empY, \
                   self.x, recov, 'b--')

            ax.plot([xAsymptote, xAsymptote], [ySymp,  ySymp],'*r', markersize = 10)
            ax.plot([xHalf, xHalf], [yHalf, yHalf], 'g*', markersize = 10)
            ax.plot([xCheapTote, xCheapTote], [yCheapTote, yCheapTote], 'b*', markersize = 10)
            print(idtsEstimated)
        # check the standard deviation of the found x for asymptote
        self.assertAlmostEqual(std(stabilityAsymptote), 0, places = 0)
        ax.legend(['True', 'noisified', 'recovered', 'Abs idt', 'rel idt'])
        MSE = ((self.trueY - recov)**2).mean()
        ax.set_title('MSE with ground truth {}'.format(MSE))
    # def tearDown(self):
    #     plotz.saveAllFigures(useLabels = False, path = 'Figures/tests')
if __name__ == '__main__':
    test = Plotting()
    test.setUp()
    test.testIDT()
    show()
