#distutils: language=c++

cdef extern from "<random>" namespace "std" nogil:
    cdef cppclass mt19937:
        mt19937() except +
        mt19937(unsigned int) except +

    cdef cppclass uniform_real_distribution[T]:
        uniform_real_distribution()
        uniform_real_distribution(float, float)
        T operator()(mt19937)

cdef class Sampler:

    def __init__(self, int seed, float low, float high):
        self.engine  = mt19937(seed)
        self.uniform = uniform_real_distribution[float](low, high)

    cdef sample(Sampler self):
        return self.uniform(self.engine)