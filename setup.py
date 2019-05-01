# from distutils.core import setup, Extension
from setuptools import setup
from setuptools.extension import Extension


# MODULE INFORMATION
__module__              = "it"
__author__              = "Casper van Elteren"
__email__               = "caspervanelteren@gmail.com"
__description__         = "General toolbox for modelling discrete systems using information theory"
__license__             = "MIT"
__python__requires__    = ">=3.6"
__version__             = '1.0'


# clang seems faster on my machine
from subprocess import run
import re, os
clangCheck = run("clang --version".split(), capture_output= True)
if not clangCheck.returncode:
    version =  re.findall('(\d+.)+', str(clangCheck.stdout))[0]
    if version <= '8.0.0':
        print("Warning older clang version used. May not compile")
    os.environ['CXXFLAGS'] = "clang++ -Xclang -fopenmp -Wall -fno-wrapv -fast-math -Ofast -std=c++17 -march=native"
    os.environ['CC']       = "clang++ -Xclang -fopenmp -Wall -fno-wrapv -ffast-math -Ofast -std=c++17 -march=native"


run('conda install -c conda-forge --file requirements.txt'.split())

from Cython.Build import cythonize
from Cython.Distutils import build_ext
import numpy, multiprocessing as mp
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
                                                 '-std=c++17',\
                                                '-fno-wrapv',\
                                                # '-g',\
                                                ],\
                           extra_link_args = ['-fopenmp',\
                                              '-std=c++17',
                                              # '-g'\
                                              ],\
            )
            exts.append(ex)


# # compile
# with open('requirements.txt', 'r') as f:
#     install_dependencies = [i.strip() for i in f.readlines()]
#
setup(\
    name            = __name__,\
    author          = __author__,\
    version         = __version__,\
    author_email    = __email__,\
    python_requires = __python__requires__,\
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
