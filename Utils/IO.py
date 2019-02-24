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
import networkx as nx
from collections import defaultdict, OrderedDict
class DataLoader(OrderedDict):
    def __init__(self, *args, **kwargs):
        """
        Data loader because hdf5 cannot use pyobjects. I am dumb because
        I want to store the graph with the data, hdf5 is probably better
        If the end-user doesn't care about this (and faster). Regardless,
        here is a hdf5 emulator.

        The format is :
        data[temperature][pulse] = SimulationResult // old format is dict
        
        THe object returns an ordered dict with the leafes consisiting of the 
        filenames belowing to the subdict categories
        """
        dataDir = args[0] if args else ''
        super(DataLoader, self).__init__(**kwargs)
        
        allowedExtensions = "pickle h5".split()
        #TODO :  make aggregate dataclass -> how to deal with multiple samples
        # current work around is bad imo
        if dataDir:
            # Warning: this only works in python 3.6+ due to how dictionaries retain order
            print("Extracting data...")
            if not dataDir.endswith('/'):
                dataDir += '/'
            files = []
            # walk root and find all the possible data files
            for root, dir, fileNames in os.walk(dataDir):
                for fileName in fileNames:
                    for extension in allowedExtensions:
                        if fileName.endswith(extension):
                            files.append(f'{root}/{fileName}')
                            break # prevent possible doubles
            files = sorted(files, key = lambda x: \
                           os.path.getctime(x))
            """
            Although dicts are ordered by default from >= py3.6
            Here I enforce the order as it matters for matching controls
            """
            data = DataLoader()
            # files still contains non-compliant pickle files, e.g. mags.pickle
            for file in files:
                # filter out non-compliant pickle files                
                try:
                    # look for t=
                    temp = re.search('t=\d+\.[0-9]+', file).group()
                    # deltas = re.search('deltas=\d+', file).group()
                    # deltas = re.search('\d+', deltas).group()
                    # look for pulse
                    pulse = re.search("\{.*\}", file).group()
                    structure = [temp]
                    if pulse == '{}':
                        structure += ['control']
                    else:
                        # there is a bug that there is a white space in my data;
                        structure += pulse[1:-1].replace(" ", "").split(':')[::-1]
                    # tmp  = loadPickle(file)
                    self.update(addData(data, file, structure))
                # attempt to load with certain properties, if not found gracdefully exit
                # this is the case for every pickle file other than the ones with props above
                except AttributeError:
                    continue 
            print('Done')

def getGraph(folderName):
    """
    Exctract the graph from the rootfolder
    and writes the graph if it is not in the folder
    """
    folderName = folderName if folderName.endswith('/') else folderName + '/'
    print('Looking for graph')
    try:
        return loadPickle(folderName + "graph.pickle")
    except FileNotFoundError:
        print("Default not found, attempting load from data")
        graph = None
        for root, dirs, files in os.walk(folderName):
            for file in files:
                try:
                    graph  = loadData(root + '/' + file).graph
                    # write graph
                    print("Graph found! writing to file")
                    savePickle(folderName + 'graph.pickle', graph)
                    return graph
                except:
                    continue
    finally:
        print("No suitable file found :(")


def loadData(fileNames):
#    print(isinstance(fileNames, str), fileNames) # debug
    if isinstance(fileNames, list):
        data = []
        for fileName in fileNames:
            data.append(loadPickle(fileName))
    elif isinstance(fileNames, str):
        data = loadPickle(fileNames)
    return data


def addData(data, toAdd, structure):
    name = structure[0]
    if len(structure) == 1:
        data[name] = data.get(name, []) + [toAdd]
        return data
    else:
        data[name] = addData(data.get(name, OrderedDict()), toAdd, structure[1:])
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

# forced rename after module structure
import io
class RenameUnpickler(pickle.Unpickler):
    # overwrite defaults
    def find_class(self, module, name):
        renamed_module = module
        # replace the name with module struct
        if module == "IO":
            renamed_module = "Utils.IO"
        # load it
        return super(RenameUnpickler, self).find_class(renamed_module, name)

def renamed_load(file_obj):
    return RenameUnpickler(file_obj).load()


def renamed_loads(pickled_bytes):
    file_obj = io.BytesIO(pickled_bytes)
    return renamed_load(file_obj)

def loadPickle(fileName):
    import sys
    with open(fileName, 'rb') as f:
        return renamed_load(f)

def savePickle(fileName, objects):
    #TODO: warning; apparantly pickle <=3 cannot handle files
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
    """
    Reads settings from target directory
    with minor backward compatibility 
    """
    for root, subdirs, files in os.walk(targetDirectory):
        for file in files:
            print(file)
            if "settings" in file:
                with open(os.path.join(root, file), 'r') as f:
                    return json.load(f)
    print("Settings not found emulating settings vector")
    """
    Backward-compatibility (legacy)
    when no setting file is present this tries to extract relevant data from the actually
    data files. 
    """
    settings    = {}
    translation = dict(k = 'repeats') # bad
    for root, subdirs, files in os.walk(targetDirectory):
        for file in files:
            # remove extension and split properties in to variables
            file = file.split(dataType)[0].split('_')
            for prop in file:
                try:
                    key, value = prop.split('=')  # filters out non-settings
                    key = translation.get(key, key)
                    settings[key] = value
                # either ValueError (not enough to split) or something else
                except:
                    continue
        if settings:
            return settings
    # loop completes; if it does it means not file was found
    else:
        print(settings)
        raise FileNotFoundError
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
