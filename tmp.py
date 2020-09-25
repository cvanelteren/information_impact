import sys; sys.path.insert(0, '../')
import networkx as nx
from plexsim.models import *
n = 10
g = nx.grid_graph([n, n])
m = Potts(g)
res = m.updateState(m.sampleNodes(1)[0])
print(res)
