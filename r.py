from fastIsing import Ising
import networkx as nx, time
graph = nx.barabasi_albert_graph(500, 200)


m = Ising(graph, 1)
s = time.time()
m.simulate(100)
print(time.time() - s)
