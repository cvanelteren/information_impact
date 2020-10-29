from libcpp.pair cimport pair

# cdef extern from "parallel-hashmap/parallel_hashmap/phmap.h":
#     cdef cppclass parallel_hash_map[T, U, HASH=*, ALLOCATOR =*]:
#         ctypedef T key_type
#         ctypedef U mapped_type
#         ctypedef pair[const, T, U] value_type

#         cppclass iterator:
#             pair[T, U] & operator*()
#             iterator operator++()
#             iterator operator--()
#             bint operator ==(iterator)
#             bint operator !=(iterator)
#         cppclass reverse_iterator:
#             pair[T, U]& operator*()
#             iterator operator++()
#             iterator operator--()
#             bint operator == (reverse_iterator)
#             bint operator != (reverse_iterator)
