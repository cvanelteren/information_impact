# #distutils: language=c++
# from cpython cimport PyObject, Py_XINCREF, Py_XDECREF
# cdef extern from *:
#     """
#     #include <Python.h>
#     #include <mutex>
#
#     std::mutex ref_mutex;
#
#     class PyObjectHolder{
#     public:
#         PyObject *ptr;
#         PyObjectHolder():ptr(nullptr){}
#         PyObjectHolder(PyObject *o):ptr(o){
#             std::lock_guard<std::mutex> guard(ref_mutex);
#             Py_XINCREF(ptr);
#         }
#         //rule of 3
#         ~PyObjectHolder(){
#             std::lock_guard<std::mutex> guard(ref_mutex);
#             Py_XDECREF(ptr);
#         }
#         PyObjectHolder(const PyObjectHolder &h):
#             PyObjectHolder(h.ptr){}
#         PyObjectHolder& operator=(const PyObjectHolder &other){
#             {
#                 std::lock_guard<std::mutex> guard(ref_mutex);
#                 Py_XDECREF(ptr);
#                 ptr=other.ptr;
#                 Py_XINCREF(ptr);
#             }
#             return *this;
#         }
#     };
#     """
#     cdef cppclass PyObjectHolder:
#         PyObjectHolder(PyObject *o) nogil
#
#
# from libcpp.vector cimport vector
# from fastIsing cimport Ising
# from cython.parallel cimport prange, threadid
#
# import networkx as nx
#
# cdef vector[PyObjectHolder] create_vectors(PyObject *o) nogil:
#     cdef vector[PyObjectHolder] vec
#     cdef int i
#     for i in range(100):
#         vec.push_back(PyObjectHolder(o))
#     return vec
#
# from cython.operator cimport dereference as deref
# def run(object o):
#     cdef PyObject *ptr=<PyObject*>o;
#     cdef int i
#     # for i in prange(10, nogil=True):
#     # for i in range(10):
#     cdef vector[PyObjectHolder] test
#     test.push_back(PyObjectHolder(<PyObject *> o))
#     <object > <PyObject> test[0]
#
#
#     # print(type(a))
#
#     # return test
# g = nx.path_graph(3)
# print(run(Ising(g, 1)))
#
#    # PyObjectHolder automatically decreases ref-counter as soon
#    # vec is out of scope, no need to take additional care
