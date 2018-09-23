# -*- coding: utf-8 -*-
"""
Created on Mon Apr 23 12:25:44 2018

@author: Cas
"""

import fastIsing,  networkx as nx
from numpy import *
from matplotlib.pyplot import *
# cluster search algorithm: pick a node; take all its NN cluster them 
class IsingSW(fastIsing.Ising):
    def __init__(self, graph, temperature = 1, doBurnin = True):
        super(IsingSW, self).__init__(graph = graph, temperature = temperature, doBurnin = doBurnin)
    
    
    def findClusters(self):
        idx      = random.choice(self.nodeIDs, size = self.nNodes, replace = False)
        clusters = zeros(self.nNodes, dtype = int)
        for node in idx:
            # if no cluster assignment add it to cluster
            if clusters[node] == 0:
                clusters = self.findAndFlipCluster(clusters, node)
        return clusters
    
    def findAndFlipCluster(self, clusters, node):
        flip = True if random.rand() < .5 else False
        q = [node] # start with a root
        while (q != []) is True:
            node = q.pop() # get node of stack
            neighbors, interactions = self.edgeData[node], self.interaction[node] # consider neighbors
#            J  = self.states[node] * self.states[neighbors].dot(interactions)
#            pAdd      = 1 - exp(- 2 * self.beta * J)
            clusters[node] = 1 # mark node as discovered
            for neighbor, interaction in zip(neighbors, interactions):
                if clusters[neighbor] != 1:
                     if self.states[neighbor] == self.states[node]:
                        pAdd = 1 - exp(- 2 * self.beta * interaction) # add node with prob
                        if random.rand() < pAdd:
                            q.append(neighbor)
                            clusters[neighbor] = 1 # discovered
#            print(q); assert 0
            if flip:
#                print(q).
                self.states[node] *= -1 
#                self.states[q] *= -1
        return clusters
        
            
    def updateState(self):
        clusters = self.findClusters()
        return self.states
if __name__ == '__main__':
    close('all')
    
    n = 32
    import fastIsing
    graph = nx.lattice.grid_2d_graph(n, n)
    temperatures = [0,1]
    print('test')
    a = fastIsing.matchTemperature(graph = graph,temperatures = temperatures, step = 1, nSamples = 100)
    t = 2.269
    assert 0 
    #%%
    #graph = nx.path_graph(n)
    model = IsingSW(graph = graph, temperature = t, doBurnin = False)
    fig, ax = subplots()
    s = model.states
    h = ax.imshow(s.reshape(n,n), vmin = -1, vmax = 1)
    while True:
        s = model.updateState()
        h.set_data(s.reshape(n,n))
        pause(.5)
#        pause(1)
    #%% 
        model2 = fastIsing.Ising(graph, temperature = t, doBurnin = False, mode = 'async' )
        fig, ax = subplots();
        s = model2.states
        h = ax.imshow(s.reshape(n,n))
    
        while True:
            s = model2.updateState()
            h.set_data(s.reshape(n,n))
            pause(1e-30)

    