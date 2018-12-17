# from distutils.core import setup, Extension
from setuptools import setup
from setuptools.extension import Extension
from Cython.Build import cythonize
from Cython.Distutils import build_ext
import numpy, os, multiprocessing as mp
# MODULE INFORMATION
__module__              = "it"
__author__              = "Casper van Elteren"
__email__               = "caspervanelteren@gmail.com"
__description__         = "General toolbox for modelling discrete systems using information theory"
__license__             = "MIT"
__python__requires__    = ">=3.6"


# clang seems faster on my machine
os.environ['CXXFLAGS'] = "clang++ -Xclang -fopenmp -Wall -fno-wrapv -fast-math -Ofast -std=c++17 -march=native"
os.environ['CC']       = "clang++ -Xclang -fopenmp -Wall -fno-wrapv -ffast-math -Ofast -std=c++17 -march=native"

exts = []

baseDir =  os.getcwd() + os.path.sep
nums = numpy.get_include()
for (root, dirs, files) in os.walk(baseDir):
    for file in files:
        fileName = f'{root}/{file}'
        if file.endswith('.pyx'):
            name = file.split('.')[0]
            #  + '/numpy'
            extPath  = fileName.replace(baseDir, '') # make relative
            extName  = extPath.split('.')[0].replace(os.path.sep, '.') # remove extension
            sources  = [extPath]
            # extName = name
            # sources = [file]
            # print(extName, sources)

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

# compile
setup(\
    name            = __name__,\
    author          = __author__,\
    author_email    = __email__,\
    python_requires = __python__requires__,\
    zip_safe        = False,\
    # cmdclass        = dict(build_ext = build_ext),\
    # packages    = ["Models", "Utils", "Toolbox"],\
    # cmdclass    = dict(build_ext = "build_ext"),\
    # gdb_debug =True,
    ext_modules = cythonize(\
            exts,\
            annotate            = True,\
            language_level      = 3,\
            compiler_directives = dict(\
                                    fast_gil       = True,\
                                    # binding      = True,\
                                    # embedsignature = True,\
                            ),\
            nthreads = mp.cpu_count(),\ # source must be pickable
            )\
)
