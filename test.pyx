# distutils: language = c++
# distutils: extra_compile_args = -std=c++11
import numpy as np
cimport numpy as np
cimport cython

@cython.auto_pickle(True)
cdef class Test:
    cdef:
        np.ndarray _states
        dict __dict__
    def __init__(self, states):
        self._states = states
        print(self._states)

    @property
    def states(self): return np.asarray(self._states)

    @states.setter
    def states(self, value):
        self._states [:] = value

    cdef double func(self, long [::1] x):
        return np.sum(x)
    # property _states:
    #     def __get__(self):
    #         return np.asarray(self._states)
    #     def __set__(self, values):
    #         cdef long[::1] self._states = self.states

import pickle
test = np.ones(10, dtype = int)
a = Test(test)
pickle.dumps(a)
