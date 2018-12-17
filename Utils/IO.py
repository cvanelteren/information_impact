#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 12:05:59 2018

@author: casper
"""

from numpy import *
from matplotlib.pyplot import *
from dataclasses import dataclass
import pickle, pandas, os, re, json, datetime
def extractData(dataDir, keys = None):
    """
    Provides a dictionary of the results
    The format is :
    data[temperature][pulse] = SimulationResult // old format is dict

    """
    #TODO :  make aggregate dataclass -> how to deal with multiple samples
    # current work around is bad imo
    data = {} # storage
    # Warning: this only works in python 3.6+ due to how dictionaries retain order
    if not dataDir.endswith('/'):
        dataDir += '/'
    filesDir = sorted(\
                     os.listdir(dataDir), \
                     key = lambda x: os.path.getctime(dataDir + x)\
                     )

    for file in filesDir:
        if file.endswith('.pickle') and 'mags' not in file:
            # look for t=
            temp   = re.search('t=\d+\.[0-9]+', file).group()

            # deltas = re.search('deltas=\d+', file).group()
            # deltas = re.search('\d+', deltas).group()

            # look for pulse
            pulse     = re.search("\{.*\}", file).group()
            tmp       = loadPickle(f'{dataDir}/{file}')
            # ensure backwards compatibility, store new format
            # should phase out over time
            if isinstance(tmp, dict):
                oldFormatConversion(dataDir, file, tmp)

            fileDict  = {pulse : [tmp]}
            # append dict
            if temp in data.keys():
                # TODO : workaround, appends samples as a list
                data[temp].update({pulse : [*data[temp].get(pulse, []), tmp]})
            else:
                data[temp] = fileDict
    return data

# TODO: needed?
def oldFormatConversion(dataDir, file, tmp):
    """
    Convert the old format to the new dataclass
    """
    tmp = SimulationResult(**tmp)
    if not os.path.exists(dataDir + '/old'):
        os.mkdir(dataDir + '/old')
    os.rename(dataDir + f'/{file}', f"{dataDir}/old/{file}") # save copy
    savePickle(dataDir + f'/{file}', tmp)


def newest(path):
    """
    Returns sorted files by time
    """
    files = os.listdir(path)
    paths = [os.path.join(path, basename) for basename in files]
    return sorted(paths, key=os.path.getctime)

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

def saveSettings(targetDirectory, settings):
    print('Saving settings')
    with open(targetDirectory + '/settings.json', 'w') as f:
        json.dump(settings, f)

def readSettings(targetDirectory, dataType = '.pickle'):
    try:
        with open(targetDirectory + '/settings.json') as f:
            return json.load(f)

    # TODO: uggly, also lacks all the entries
    except FileNotFoundError:
        # use re to extract settings
        settings = {}
        for file in os.listdir(targetDirectory):
            if file.endswith(dataType) and 'mags' not in file:
                file = file.split(dataType)[0].split('_')
                for f in file:
                    tmp = f.split('=')
                    if len(tmp) > 1:
                        key, value = tmp
                        if key in 'nSamples k step deltas':
                            if key == 'k':
                                key = 'repeats'
                            settings[key] = value
            # assumption is that all the files have the same info
            break
        saveSettings(targetDirectory, settings)
        return settings

def readCSV(fileName, **kwargs):
    '''
    :fileName: name of the file to be loaded
    :kwargs: input to pandas.read_csv (dict)
    '''
    return pandas.read_csv(fileName, **kwargs)


# TODO: make  this separate file?
@dataclass
class SimulationResult:
    """
    Standard format of collected data
    """
    conditional : dict
    px          : dict
    snapshots   : dict
    mi          : array
    # model       : object
    graph       : object
    # TODO: add these?
    # temperature : int
    # pulse       : dict
