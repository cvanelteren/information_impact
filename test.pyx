#distutils: language=c++
from cpython cimport PyObject, Py_XINCREF, Py_XDECREF
cdef extern from *:
    """
    #include <Python.h>
    #include <mutex>

    std::mutex ref_mutex;

    class PyObjectHolder{
    public:
        PyObject *ptr;
        PyObjectHolder():ptr(nullptr){}
        PyObjectHolder(PyObject *o):ptr(o){
            std::lock_guard<std::mutex> guard(ref_mutex);
            Py_XINCREF(ptr);
        }
        //rule of 3
        ~PyObjectHolder(){
            std::lock_guard<std::mutex> guard(ref_mutex);
            Py_XDECREF(ptr);
        }
        PyObjectHolder(const PyObjectHolder &h):
            PyObjectHolder(h.ptr){}
        PyObjectHolder& operator=(const PyObjectHolder &other){
            {
                std::lock_guard<std::mutex> guard(ref_mutex);
                Py_XDECREF(ptr);
                ptr=other.ptr;
                Py_XINCREF(ptr);
            }
            return *this;

        }
    };
    """
    cdef cppclass PyObjectHolder:
        PyObject *ptr
        PyObjectHolder(PyObject *o) nogil


import numpy as np
cimport numpy as np
from libcpp.vector cimport vector
from fastIsing cimport Ising
from models cimport Model
from cython.parallel cimport prange, threadid
from libc.stdio cimport printf
import networkx as nx, copy
g = nx.path_graph(3)
cdef Ising m = Ising(g,10)
cdef Ising n = copy.deepcopy(m)
# cdef PyObject *ptr = <PyObject *>m
cdef list models = []
cdef Model tmp
cdef int i
cdef vector[PyObjectHolder] vec
for i in range(10):
    tmp = Ising(g, 1)
    models.append(tmp)
    vec.push_back(PyObjectHolder(<PyObject *> tmp))

(<Ising> vec[0].ptr).simulate(100)
(<Model> vec[0].ptr).states[0] = -100
print((<Model> vec[0].ptr).states.base, models[0].states.base, models[1].states.base)
