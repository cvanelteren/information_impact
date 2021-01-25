#!/usr/bin/env python3

from os.path import dirname, basename, isfile, join
import glob, importlib

modules = glob.glob(join(dirname(__file__), "*.py"))
__all__ = [basename(f).split(".py")[0] for f in modules if isfile(f) and not f.endswith('__init__.py')]

__dict__ = {name :
            importlib.import_module(f"experiments.{name}") for name in __all__}
 
def get(key):
    return __dict__.get(key)
