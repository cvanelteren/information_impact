#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 19 10:57:39 2018

@author: casper
"""

from numpy import *
from matplotlib.pyplot import *
import testFunctions
if __name__ == '__main__':
    import unittest
    suite = unittest.TestLoader().discover('.', pattern = "test*.py")
    unittest.TextTestRunner(verbosity=0).run(suite)
    from plotting import saveAllFigures
    saveAllFigures(useLabels = False, path = 'Figures')
