from libcpp.vector cimport vector
cdef extern from "unstructured.h" nogil:
    struct container_hash:
        pass
    ctypedef intVector
    ctypedef statemap
