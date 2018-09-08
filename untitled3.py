import h5py, os
from numpy import *


fileName = 'remove.h5'
try: os.remove(fileName)
except: pass

a = {i : random.rand() for i in range(10)}
with h5py.File(fileName) as f:
    data = f.create_group('joint')
    for key, value in a.items():
        if str(key) in data:
            data[str(key)] += value
        else:
            data.create_dataset(str(key), data = value, dtype = type(value))
    a[11] = random.rand()      
    for key, value in a.items():
        print(data)
        print(key, value)
        if str(key) in data:
            add = data[key].value
            del  f[f'joint/{key}']
            data.create_dataset(str(key), data = add + value)
        else:
            data.create_dataset(str(key), data = value)
