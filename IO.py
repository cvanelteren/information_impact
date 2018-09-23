#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 12:05:59 2018

@author: casper
"""

from numpy import *
from matplotlib.pyplot import *
import pickle, pandas
def loadPickle(fileName):
    with open(fileName, 'rb') as f:
        return pickle.load(f)
def savePickle(fileName, objects):
    #TODO: warning; appearantly pickle <=3 cannot handle files
    # larger than 4 gb.
    if not fileName.endswith('.pickle'):
        fileName += '.pickle'
    print(f'Saving {fileName}')
    with open(fileName, 'wb') as f:
        pickle.dump(objects, f, protocol = pickle.HIGHEST_PROTOCOL)

def readCSV(fileName, **kwargs):
    '''
    :fileName: name of the file to be loaded
    :kwargs: input to pandas.read_csv (dict)
    '''
    return pandas.read_csv(fileName, **kwargs)
