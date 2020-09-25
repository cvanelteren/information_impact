import sys; sys.path.insert(0, '../')
from PlexSim.plexsim.models import *

def test():
    m=Ising(nx.path_graph(3))
    m.simulate(100)
    