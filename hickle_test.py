import hickle, numpy as np, networkx as nx, h5py as h5, pickle


class Test(dict):
    def __init__(self, *args, **kwargs):
        super(Test, self).__init__(**kwargs)

    def __getitem__(self, key):
        val = dict.__getitem__(self, key)
        print(f'> {type(val)}')
        return val

t = Test(test = dict(test1 = 3), d = 3)
for kk in t:
    print(t[kk])
