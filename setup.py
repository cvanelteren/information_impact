from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy
import os
os.environ['CXXFLAGS'] = "clang++ -Xclang -fopenmp -fno-wrapv"
os.environ['CC'] = "clang++ -Xclang -fopenmp -fno-wrapv"

exts = []
nums = numpy.get_include()
for file in os.listdir(os.getcwd()):
    if file.endswith('.pyx'):
        name = file.split('.')[0]
        #  + '/numpy'
        sources = [file]
        # if 'fastIsing' in file:
        #     sources.append('vfastexp.h')
        ex = Extension(name, sources = sources, \
                       include_dirs =[nums],\
                       extra_compile_args = ['-fopenmp',\
                                             '-ffast-math','-Ofast', \
                                             '-march=native',\
                                             '-std=c++11',\
                                            '-fno-wrapv',\
                                            # '-g',\
                                            ],\
                       extra_link_args = ['-fopenmp',\
                                          "-std=c++11",
                                          # '-g'\
                                          ],\
        )
        exts.append(ex)

# compile
setup(\
ext_modules = cythonize(exts,\
            annotate            = True,\
            language_level      = 3,\
            compiler_directives = dict(\
                                        fast_gil = True,\
                                       binding  = True),\
            # gdb_debug=True,
            )\
)
