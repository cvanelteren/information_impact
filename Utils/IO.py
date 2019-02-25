#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 12:05:59 2018

@author: casper
"""

from numpy import *
from matplotlib.pyplot import *
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
    try:
        print(f'Looking for graph in {folderName}')
        graph = readSettings(folderName)[0].get('graph', None)
        if graph:
            print('Graph found in settings')
            return nx.readwrite.json_graph.node_link_graph(graph)
    except FileNotFoundError:
        print("Default not found, attempting load from data")

    graph = None
    for root, dirs, files in os.walk(folderName):
        for file in files:
            print(file)
            # settings will always be read first when walked over folders
            if 'settings' in file:
                print('reading settings...')
                settings = readSettings(root)[0]
                if 'graph' in settings:
                    print('Graph found in settings')
                    return nx.readwrite.node_link_graph(settings['graph'])
            else:
                try:
                    graph  = loadData(os.path.join(root, file)).graph
                    if graph:
                        print("Graph found in data file!")
                        return graph
                except AttributeError:
                    continue
    print("No suitable file found :(")


def loadData(filename):
    # TODO: change
    if filename.endswith('pickle'):
        return loadPickle(filename)
#    print(isinstance(fileNames, str), fileNames) # debug
#    if isinstance(fileNames, list):
#        data = []
#        for fileName in fileNames:
#            data.append(loadPickle(fileName))
#    elif isinstance(fileNames, str):
#        data = loadPickle(fileNames)
#    return data


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


def readCSV(fileName, **kwargs):
    '''
    :fileName: name of the file to be loaded
    :kwargs: input to pandas.read_csv (dict)
    '''
    return pandas.read_csv(fileName, **kwargs)


# TODO: make  this separate file?
from dataclasses import dataclass, fields, field, _MISSING_TYPE
@dataclass
class Settings:
    # simulation parameters
    repeat        : int
    deltas        : int
    nSamples      : int
    step          : int
    burninSamples : int
    nNodes        : int
    nTrials       : int
    pulseSizes    : list

    # model properties
    updateMethod  : str
    # legacy support needed ofr these
    mapping       : dict 
    rmapping      : dict      

    graph         : dict  
    
    # defaults
    model         : str   = field(default = 'Models.fastIsing.Ising')
    _graph        : dict  = field(init = False, repr = False, default_factory = dict)
    directory     : str = field(init = False)  # "data directory" 
    def __init__(self, data = None):
        """
        Input:
            :data: either dict or string. If dict it assigns keys as fields. 
            If string it excpects a root folder which it will search for settings 
            or emulate them
        """
        # assign defaults
        for field in fields(self):
            if not any(isinstance(field.default, i) for i in [_MISSING_TYPE, property]):
                print(f"Setting {field.name}")
                self.__setattr__(field.name, field.default)
        # load dict keys
        if isinstance(data, dict):
            self.addItems(data)
            
        # read json file
        elif isinstance(data, str):
            self.directory = data
            self.read(data)
            
            
    
    @property
    def graph(self):
        return self._graph
    @graph.setter
    def graph(self, val):
        """
        Attempts to load the graph from a random data file
        """
        if isinstance(val, property):
            for root, subdirs, files in os.walk(self.directory):
                for file in files:
                    try:
                        graph  = loadData(os.path.join(root, file)).graph
                        if graph:
                            print("Graph found in data file!")
                            self._graph = nx.readwrite.json_graph.node_link_data(graph)
                            return 
                    except AttributeError:
                        continue
            else:
                raise ValueError("Input graph not recognized")
        elif isinstance(val, dict):
            self._graph = val
        
        
            
            # legacy support needed ofr these
    def __repr__(self):
        """
        Print all settings
        """
        banner = '-' * 32
        top    = f'\n{banner} Simulation Settings {banner}'
        
        s = top
        for k, v in self.__dict__.items():
            if not k.startswith('_'):
                s += f'\n{k:<15} = {v}' # basic alignment
        s += '\n'
        s += '-' * len(top)
        return s
    def addItems(self, data):
        """
        Helper function for read; gives warning for unintended items
        """
        for field in fields(self):
            fetched = data.get(field.name, field.default)
            # ignore missing types and privates
            if not isinstance(fetched, _MISSING_TYPE) and not field.name.startswith('_'):
                self.__setattr__(field.name, fetched)
            else:
                print(f"WARNING: {field.name.replace('_','')} not found in input")

    def read(self, targetDirectory):
        """
        Reads json settings from target directory
        with minor backward compatibility
        """
        print("Reading settings: ", end = '')
        for root, subdirs, files in os.walk(targetDirectory):
            for file in files:
                if "settings" in file:
                    print("using json")
                    with open(os.path.join(root, file), 'r') as f:
                        self.addItems(json.load(f))
                    return 0 # exit function
        print("Settings not found emulating settings vector")
        
        """
        Backward-compatibility (legacy)
        when no setting file is present this tries to extract relevant data 
        from the actually data files.
        """
    
        translation = dict(k = 'repeats') # bad
        for root, subdirs, files in os.walk(targetDirectory):
            for file in files:
                settings = {}
                # remove extension and split properties in to variables
                file = file.split(dataType)[0].split('_')
                for prop in file:
                    try:
                        key, value = prop.split('=')  # filters out non-settings
                        key = translation.get(key, key)
                        settings[key] = value
                    # either ValueError (not enough to split) or something else
                    except ValueError:
                        continue
                # when found exit
                if settings:
                    self.addItems(settings)
                    return 0
        else:
            raise FileNotFoundError
            
    def save(self, targetDirectory):
        print('Saving settings')
        s = {key.replace('_', '') : v for k, v in self.__dict__.items()}
        with open(os.path.join(targetDirectory, 'settings.json'), 'w') as f:
            json.dump(\
                      s, f,\
                      default = lambda x : float(x), \
                      indent  = 4) 
            
@dataclass
class SimulationResult(object):
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

if __name__ == '__main__':
    dp = '/home/casper/projects/information_impact/Data/'
    psycho   = '1548025318.5751357'
    settings = Settings(os.path.join(dataPath, psycho))
    print(settings)