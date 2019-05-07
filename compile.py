from Cython.Build import cythonize
from Cython.Distutils import build_ext
from setuptools import setup
from setuptools.extension import Extension
import numpy, multiprocessing as mp, os

import re, os
from subprocess import run
clangCheck = run("clang --version".split(), capture_output= True)
if not clangCheck.returncode:
    os.environ['CXXFLAGS'] = "clang++ -Xclang -fopenmp -Wall -fno-wrapv -fast-math -Ofast -std=c++17 -march=native"
    os.environ['CC']       = "clang++ -Xclang -fopenmp -Wall -fno-wrapv -ffast-math -Ofast -std=c++17 -march=native"
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
                                                 '-ffast-math','-Ofast', \
                                                 '-march=native',\
                                                 '-std=c++11',\
                                                '-fno-wrapv',\
                                                # '-g',\
                                                ],\
                           extra_link_args = ['-fopenmp',\
                                              '-std=c++11',\
                                              '-lomp',\
                                              # '-g'\
                                              ],\
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
                                    # binding      = True,\
                                    # embedsignature = True,\
                                    ),\
            # source must be pickable
            nthreads            = mp.cpu_count(),\
            ),\
# gdb_debug =True,
)


# setup(\
#     name            = __name__,\
#     author          = __author__,\
#     version         = __version__,\
#     author_email    = __email__,\
#     python_requires = __python__requires__,\
#     zip_safe        = False,\
#     ext_modules = cythonize(\
#             exts,\
#             # annotate            = True,\ # set to true for performance html
#             language_level      = 3,\
#             compiler_directives = dict(\
#                                     fast_gil       = True,\
#                                     # binding      = True,\
#                                     # embedsignature = True,\
#                                     ),\
#             # source must be pickable
#             nthreads            = mp.cpu_count(),\
#             ),\
# # gdb_debug =True,
# )
