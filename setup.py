from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy
import os

exts = []
nums = numpy.get_include()
for file in os.listdir(os.getcwd()):
    if file.endswith('.pyx'):
        name = file.split('.')[0]
        #  + '/numpy'
        ex = Extension(name, sources = [file], \
                       include_dirs =[nums],\
                       extra_compile_args = ['-fopenmp', '-Ofast', '-march=native'],\
                       extra_link_args = ['-fopenmp'],\
        )
        exts.append(ex)

# compile
setup(\
ext_modules = cythonize(exts,\
            annotate = True,\
            language_level = 3,\
            compiler_directives = dict(fast_gil = True,\
                                       binding = True))\
)
