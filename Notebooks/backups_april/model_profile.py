import sys; sys.path.insert(0, '../')
from PlexSim.plexsim.models import Ising, nx 

def test():
    m=Ising(nx.grid_2d_graph(128, 128))
    m.simulate(100)

if __name__ == '__main__':
    print("starting")
    m = Ising(nx.grid_2d_graph(128, 128))
    m.simulate(10)
