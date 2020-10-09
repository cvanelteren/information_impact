from Cython.Build import cythonize
from Cython.Distutils import build_ext
from Cython import Build
from setuptools import setup
from setuptools.extension import Extension
import numpy, multiprocessing as mp, os

import re, os
from subprocess import run
add = []
compiler = 'g++'
optFlag = '-Ofast'
cppv    = '17'

flags = f'{optFlag} -march=native -std=c++{cppv} -flto '\
        '-frename-registers -funroll-loops -fno-wrapv '\
        '-fopenmp-simd -fopenmp'
try:
    clangCheck = run(f"{compiler} --version".split(), capture_output= True)
    if not clangCheck.returncode and 'fs4' not in os.uname().nodename:
        print("Using default")
        os.environ['CXXFLAGS'] =  f'{compiler} {flags}'
        # os.environ['CC']       =  f'{compiler} {flags}'
        # add.append('-lomp') # c
except Exception as e:
    print(e)
    pass
# collect pyx files
exts = []
# baseDir =  os.getcwd() + os.path.sep +

BASE = "imi"
nums = numpy.get_include()

for (root, dirs, files) in os.walk(BASE):
    for file in files:
        fileName = os.path.join(root, file)
        if file.endswith('.pyx') and not "test" in fileName:
            # some cython shenanigans
            # extPath  = fileName.replace(baseDir, '') # make relative
            extPath = os.path.join(root, file)
            extName, exension  = os.path.splitext(extPath.replace(os.path.sep, '.'))# remove extension
            sources  = [extPath]
            ex = Extension(extName, \
                           sources            = sources, \
                           include_dirs       = [nums, '.'],\
                           extra_compile_args = flags.split(),\
                           extra_link_args = ['-fopenmp',\
                                              f'-std=c++{cppv}',\
                                              # '-g'\
                                              ] + add,\
                            define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
            )
            exts.append(ex)
cdirectives =  dict(\
                    fast_gil         = True,\
                    boundscheck      = False,\
                    cdivision        = True,\
                    initializedcheck = False,\
                    overflowcheck    = False,\
                    nonecheck        = False,\
                    binding          = True,\
                    # embedsignature = True,\
                    )

from setuptools import find_packages, find_namespace_packages

__version__ = 2.0
packages     = find_packages(where = BASE,
                             include = ["utils*"])
# assert 0
setup(\
        name         = 'imi',
        version      = __version__,
        author       = "Casper van Elteren",
        author_email = "caspervanelteren@gmail.com",
        url          = "cvanelteren.github.io",
        zip_safe     = False,
        package_dir  = {"" : BASE},
        package_data = dict(infcy = '*.pxd'.split()),\
        cmdclass = {'build_ext': Build.build_ext},
        ext_modules  = cythonize(
                exts,
                language_level      = 3,
                compiler_directives = cdirectives,
                nthreads            = mp.cpu_count(),
                ),\

)
