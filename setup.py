 # distutils: language=c
from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy
import os

for file in os.listdir(os.getcwd()):
    if file.endswith('.pyx'):
        print(f'{file:^}')
        f = numpy.get_include() + '/numpy'
        ext = cythonize(\
                    file, annotate = True,
                    language_level = 3, \
                    gdb_debug = True,\
                    include_path = [numpy.get_include()])
        setup(\
        ext_modules = ext,\
        extra_compile_args = ['-openmp'],\
        extra_link_args    = ['-openmp'],\
                    #extra_compile_args = ['-std=c++11'],\
        # include_paths= [f],\
        include_dirs =[numpy.get_include()]\
        )
        print('Done')



# ext_modules = [
#  Extension('cy', ['cy.pyx'], extra_compile_args=['-fopenmp'], extra_link_args = ['-fopenmp'])
#
# ]
# setup(ext_modules = cythonize(ext_modules))
