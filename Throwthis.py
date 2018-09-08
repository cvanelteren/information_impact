import subprocess, time


s = time.time()
n = 109
m = 2
t =int(1e6) * n * m
subprocess.call(f'./testThis {n} {m} {t}'.split(' '))
# print(time.time() - s)
#
# import fastIsing, networkx as nx
#
# ss = time.time()
# graph = nx.grid_2d_graph(n,n, periodic = True)
# print(time.time() - ss)
# ss = time.time()
# model = fastIsing.Ising(graph, 20, False, mode = 'single')
# s = time.time()
# model.simulate(nn, 1)
# print(time.time() - s, time.time() - ss)
