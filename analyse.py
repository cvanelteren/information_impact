import numpy as np, matplotlib.pyplot as plt, plotting as plotz,\
information, os, IO, fastIsing, networkx as nx


aDir = f'{os.getcwd()}/Data/'
bDir = f'{os.getcwd()}/Psycho/'
dataDir = 'Psycho' # relative path careful
df    = IO.readCSV(f'{bDir}/Graph_min1_1.csv', header = 0, index_col = 0)
h     = IO.readCSV(f'{bDir}/External_min1_1.csv', header = 0, index_col = 0)

graph   = nx.from_pandas_adjacency(df) # weights done, but needs to remap to J or adjust in Ising
model   = fastIsing.Ising(graph, 1, False)

colors = plt.cm.tab20(np.arange(model.nNodes))
for file in os.listdir(aDir):
    if file.endswith('pickle'):
        res = IO.loadPickle(aDir + file)
        fig, ax = plt.subplots()
        I = res['I']
        [ax.plot(i, '.', color = c) for i, c in zip(I.T, colors)]
        ax.set_title(file)
plt.show()
