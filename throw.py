from fastIsing import Ising
import networkx as nx
from numpy import *
from matplotlib.pyplot import *
graph = nx.barabasi_albert_graph(10,1)
model = Ising(graph = graph, temperature = 2, \
              mode = 'async', doBurnin = False)
#import sys
#
#def trace(frame, event, arg):
#    print ("%s, %s:%d" % (event, frame.f_code.co_filename, frame.f_lineno))
#    return trace
##sys.settrace(trace)
#model.simulate(10)
#import infcy
#from time import time
#s= time()
#n, nn = 100, 100
#for i in range(n):
#    model.sampleNodes(nn)
dataDir = 'Psycho' # relative path careful
import IO
df    = IO.readCSV(f'{dataDir}/Graph_min1_1.csv', header = 0, index_col = 0)
h     = IO.readCSV(f'{dataDir}/External_min1_1.csv', header = 0, index_col = 0)
 #
graph   = nx.from_pandas_adjacency(df)
for i, j in graph.edges():
     graph[i][j]['weight'] = 1

attr = {}
for node, row in h.iterrows():
    attr[node] = dict(H = row['externalField'], nudges = 0)
nx.set_node_attributes(graph, attr)
import models
m = models.Model(graph = graph, agentStates = [-1, 1])
print(m.graph)
p = m.mapping
m.simulate(pulse = p)

#print(time() - s)
#
#s = time()
#for i in range(n):
#    [random.choice(model.nodeIDs, size = model.nNodes, replace = 0) \
#                   for _ in range(nn)]
#print(time()- s)



#print(snapshots, sum(list(snapshots.values())))
##model.states.fill(1)
#n = 100
#xs = []
#for t in linspace(0, 5, n):
#    model.t =  t
#    model.states.fill(1)
#    x = abs(model.simulate(100).mean())
#    xs.append(x)
#    print(model.sampleNodes[model.mode](model.nodeIDs))
#plot(xs)
#
