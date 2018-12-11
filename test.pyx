# distutils: language=c++

from cpython cimport PyObject,Py_XINCREF, Py_XDECREF
cdef extern from *:
    """
    #include <Python.h>
    class PyObjectHolder{
    public:
        PyObject *ptr;
        PyObjectHolder():ptr(nullptr){}
        PyObjectHolder(PyObject *o):ptr(o){
           Py_XINCREF(ptr);
        }
        //rule of 3
        ~PyObjectHolder(){
            Py_XDECREF(ptr);
        }
        PyObjectHolder(const PyObjectHolder &h):
            PyObjectHolder(h.ptr){}
        PyObjectHolder& operator=(const PyObjectHolder &other){
            Py_XDECREF(ptr);
            ptr=other.ptr;
            Py_XINCREF(ptr);
            return *this;
        }
    };
    """
    cdef cppclass PyObjectHolder:
        PyObjectHolder(PyObject *o)
from fastIsing cimport Ising
from models cimport Model
from libcpp.vector cimport vector
from libc.stdio cimport printf
from cython.operator cimport dereference as deref
import networkx as nx
import numpy as np
cimport numpy as np
import copy
cimport cython
@cython.auto_pickle(True)
cdef class Temp:
    cdef dict __dict__
    cdef int N
    cdef np.ndarray a
    def __init__(self, a):
        self.a = a
        self.N = a.size

    cdef double test(self) nogil:
        cdef int i
        cdef double[::1] a = self.a
        cdef double mu
        cdef double* ptr
        for i in range(self.N):
            mu += a[i]
        return mu
x = np.ones(10)
cdef Temp a = Temp(x)
print(a.a)
cdef double y = a.test()
print(y)
    # def __getstate__(self):
    #     return (self.a)
    # def __setstate__(self, state):
    #     print(id(state))
    #     self._a = state
