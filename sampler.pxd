# distutils: language=c++
cdef extern from "<random>" namespace "std" :
    cdef cppclass mt19937:
        mt19937() except +
        mt19937(unsigned int) except +

    cdef cppclass uniform_real_distribution[T]:
        uniform_real_distribution()
        uniform_real_distribution(float, float)
        T operator()(mt19937)


cdef class Sampler:
    cdef:
        mt19937 engine
        uniform_real_distribution[float] uniform
    cdef sample(Sampler self)
