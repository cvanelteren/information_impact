import numpy as np
import matplotlib.pyplot as plt
import h5py as h
import os
from Utils import IO

class C:
    def __init__(self):
        self._x = 1
    @property
    def x(self): return self._x
    
    @x.setter
    def x(self, v):
        self._x = v
        

dataPath = f"{os.getcwd()}/Data/"

c = C()
print(type(c.x))
print(settings)