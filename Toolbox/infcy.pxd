# from libcpp.vector cimport vector
# from libcpp.pair   cimport pair
# from libcpp.unordered_map cimport unordered_map
# cdef extern from "unstructured.h" nogil:
#     struct container_hash:
#         const size_t operator()(Container)
#
#     cdef cppclass statemap[T, U, C]:
#         ctypedef T key_type
#         ctypedef U mapped_type
#         ctypedef C hash_function
#
#         # operator definitions
#         cppclass iterator:
#             pair[T, U]& operator*()
#             iterator operator++()
#             iterator operator--()
#             bint operator==(iterator)
#             bint operator !=(iterator)
#
#         cppclass reverse_iterator:
#             pair[T, U]& operator*()
#             iterator operator++()
#             iterator operator--()
#             bint operator==(reverse_iterator)
#             bint operator!=(reverse_iterator)
#
#         cppclass const_iterator(iterator):
#             pass
#         cppclass const_reverse_iterator(reverse_iterator):
#             pass
#
#         unordered_map() except +
#         unordered_map(unordered_map&) except +
#
#         U& operator[](T&)
#
#         # class operators
#         bint operator==(unordered_map&, unordered_map&)
#         bint operator!=(unordered_map&, unordered_map&)
#         bint operator<(unordered_map&, unordered_map&)
#         bint operator>(unordered_map&, unordered_map&)
#         bint operator>=(unordered_map&, unordered_map&)
#         bint operator<=(unordered_map&, unordered_map&)
#
#         U& at(const T&)
#         const U& const_at "at"(const T&)
#
#         # define class methods?
#         iterator begin()
#         const_iterator const_begin "begin"()
#
#         void clear()
#
#         size_t count(T&)
#
#         bint empty ()
#
#         iterator end()
#
#         const_iterator const_end "end"()
#         pair[iterator, iterator] equal_range(T&)
#         pair[const_iterator, const_iterator] const_equal_range "equal_range"(const T&)
#
#         iterator erase(iterator)
#         iterator erase(iterator, iterator)
#
#         size_t erase(T&)
#         iterator find(T&)
#
#         const_iterator const_find "find"(T&)
#
#         pair[iterator, bint] insert(pair[T,U])
#         iterator insert(iterator, pair[T, U])
#         iterator insert(iterator, iterator)
#
#         iterator lower_bound(T&)
#         const_iterator const_lower_bound "lower_bound"(T&)
#
#         size_t max_size()
#         reverse_iterator rbegin()
#         const_reverse_iterator const_rbegin "rbegin"()
#
#         reverse_iterator rend()
#         const_reverse_iterator const_rend "rend"()
#
#         size_t size()
#         void swap(unordered_map&)
#
#         iterator upper_bound(T&)
#         const_iterator_upper_bound "upper_bound"(T&)
#
#         void max_load_factor(float)
#         float max_load_factor()
#
#         void rehash(size_t)
#
#         void reserve(size_t)
#
#         size_t bucket_count()
#         size_t max_bucket_count()
#         size_t bucket_size(size_t)
#         size_t bucket(const T&)
