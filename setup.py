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

run('conda install -c conda-forge --file requirements.txt'.split())
import compile
