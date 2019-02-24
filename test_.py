import numpy as np
import matplotlib.pyplot as plt
import h5py as h
import deepdish as dd

from Utils import IO


import networkx as nx

#dataDir = 'Psycho' # relative path careful
#df    = IO.readCSV(f'{dataDir}/Graph_min1_1.csv', header = 0, index_col = 0)
#h     = IO.readCSV(f'{dataDir}/External_min1_1.csv', header = 0, index_col = 0)
#graph   = nx.from_pandas_adjacency(df)
#attr = {}
#for node, row in h.iterrows():
#    attr[node] = dict(H = row['externalField'], nudges = 0)
#    nx.set_node_attributes(graph, attr)
#nx.write_gml(graph, "test.gz")


def tmp():
    try:
        a = 0 + 1
        try:
            b = 1
            print(a, b)
        except:
            pass
    except:
        pass
    return None
tmp()