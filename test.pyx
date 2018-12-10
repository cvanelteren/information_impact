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

cdef class Temp:
    cdef int a
    def __init__(self, a):
        self.a = 1
    cdef double rand(self):
        return 5.
from fastIsing cimport Ising
from models cimport Model
from libcpp.vector cimport vector
from libc.stdio cimport printf
from cython.operator cimport dereference as deref
import networkx as nx
import numpy as np
cimport numpy as np

cdef int i, j, k, N = 10

g =  nx.path_graph(30)
cdef Ising m = Ising(g, 1)
from cython.parallel cimport prange
cdef double func(Model m, int deltas)nogil:
    with gil:
        print(id(m))
    cdef int i
    for i in range(deltas):
        m.sampleNodes(1)
    return 0

for i in prange(N, nogil = 1):
    func(m, 10)
