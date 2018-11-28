 # distutils: language=c++
from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy
import os

for file in os.listdir(os.getcwd()):
    if file.endswith('.pyx'):

        f = numpy.get_include() + '/numpy'
        setup(\
        ext_modules = cythonize(\
                    file, annotate = True,
                    language_level = 3, \
                    gdb_debug = True, language = 'c++'),\
                    extra_compile_args = ['-std=c++11'],\
        include_dirs =[numpy.get_include()]\
        )


# ext_modules = [
#  Extension('cy', ['cy.pyx'], extra_compile_args=['-fopenmp'], extra_link_args = ['-fopenmp'])
#
# ]
# setup(ext_modules = cythonize(ext_modules))
