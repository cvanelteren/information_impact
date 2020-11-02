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
        '-fopenmp-simd -fopenmp -Wfatal-errors'
# collect pyx files
exts = []
# baseDir =  os.getcwd() + os.path.sep +

BASE = "imi"
nums = numpy.get_include()
for (root, dirs, files) in os.walk(BASE):
    for file in files:
        fileName = os.path.join(root, file)
        if file.endswith('.pyx'):
            # some cython shenanigans
            # extPath  = fileName.replace(baseDir, '') # make relative
            extPath = os.path.join(root, file)
            extName, extension  = os.path.splitext(extPath.replace(os.path.sep, '.'))# remove extension

            # extName = extName.replace("core.", '')

            sources  = [extPath]
            ex = Extension(extName,
                           sources            = sources,
                           include_dirs       = [nums, '.'],
                           extra_compile_args = flags.split(),
                           language = "c++",
                           extra_link_args = ['-fopenmp',
                                              f'-std=c++{cppv}',
                                              # '-g'
                                              ] + add,
                            define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
            )
            exts.append(ex)
def find_pxd(base) -> list:
    """
    package pxd files
    """
    data_files = []
    for root, dirs, files in os.walk(base):
        for file in files:
            if file.endswith('pxd'):
                # base     = os.path.basename(base)
                file = os.path.join(root, file)
                print(root, file)
                data_files.append([root, [file]])

    return data_files
data_files = find_pxd("imi")
cdirectives =  dict(
                    boundscheck      = False,
                    cdivision        = True,
                    initializedcheck = False,
                    overflowcheck    = False,
                    nonecheck        = False,
                    binding          = True,
                    # embedsignature = True,
                    )

from setuptools import find_packages, find_namespace_packages
__version__ = 2.0
pbase = "imi"
# todo: make prettier

package_data = {"" : "*.pxd *.pyx".split(),
                # "imi.core" : "*.pxd *.pyx".split(),
                }
packages = find_packages(exclude = ["Plexsim*"])
setup(
        name         = 'imi',
        version      = __version__,
        author       = "Casper van Elteren",
        author_email = "caspervanelteren@gmail.com",
        url          = "cvanelteren.github.io",
        zip_safe     = False,
        # namespace_packages     = namespaces,
        include_package_data = True,
        data_files   = data_files,
        # package_data = package_data,
        packages     = packages,
        ext_modules  = cythonize(
                exts,
                compiler_directives = cdirectives,
                nthreads            = mp.cpu_count(),
                ),

)
