from Cython.Build import cythonize
from Cython.Distutils import build_ext
from setuptools import setup
from setuptools.extension import Extension
import numpy, multiprocessing as mp, os

import re, os
from subprocess import run
add = []
try:
    compiler = 'zapcc++'
    OPTIMIZATION = '-O2'
    clangCheck = run(f"{compiler} --version".split(), capture_output= True)
    if not clangCheck.returncode and 'fs4' not in os.uname().nodename:
        print("Using default")
        os.environ['CXXFLAGS'] = f"{compiler} -Xclang -fopenmp -Wall -fno-wrapv -std=c++11 {OPTIMIZATION} -march=native"
        os.environ['CC']       = f"{compiler} -Xclang -fopenmp -Wall -fno-wrapv -std=c++11 {OPTIMIZATION}  -march=native"
        add.append('-lomp') # clang openmp stuff
except Exception as e:
    print(e)
    pass
# collect pyx files
exts = []
baseDir =  os.getcwd() + os.path.sep
nums = numpy.get_include()
if not os.path.exists('Data'):
    os.mkdir('Data')
for (root, dirs, files) in os.walk(baseDir):
    for file in files:
        fileName = f'{root}/{file}'
        if file.endswith('.pyx'):
            # some cython shenanigans
            extPath  = fileName.replace(baseDir, '') # make relative
            extName  = extPath.split('.')[0].replace(os.path.sep, '.') # remove extension
            sources  = [extPath]
            ex = Extension(extName, \
                           sources            = sources, \
                           include_dirs       = [nums, '.'],\
                           extra_compile_args = ['-fopenmp',\
                                                 '-march=native',\
                                                 '-std=c++17',\
                                                '-fno-wrapv',\
                                                 OPTIMIZATION,\
                                                # '-g',\
                                                ],\
                           extra_link_args = ['-fopenmp',\
                                              '-std=c++17',\
                                              # '-g'\
                                              ] + add,\
            )
            exts.append(ex)
# # compile
# with open('requirements.txt', 'r') as f:
#     install_dependencies = [i.strip() for i in f.readlines()]
#
setup(\
    zip_safe        = False,\
    ext_modules = cythonize(\
                            exts,\
                            # annotate            = True,\ # set to true for performance html
                            language_level      = 3,\
                            compiler_directives = dict(\
                                         fast_gil       = True,\
                                         boundscheck    = False,\
                                         cdivision      = True,\
                                         initializedcheck = False,\
                                         overflowcheck  = False,\
                                         nonecheck      = False,\
                                         binding        = False,\
                                         # binding      = True,\
                                         # embedsignature = True,\
              ),\
                            # source must be pickable
                            nthreads            = mp.cpu_count(),\
    ),\
# gdb_debug =True,
)

