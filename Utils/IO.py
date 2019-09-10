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
def loadSets(data : dict) -> dict:
    """
    Load a set of data. Returns the settings and the loaded data
    """
    from multiprocessing.pool import ThreadPool
    import multiprocessing as mp
    settings   = {}
    loadedData = {}

    pbar = pr.ProgBar(len(data))
    for idx, (k, v) in enumerate(data.items()):
        # tmp = os.path.join(root, k)
        print(f'Loading sets {k}')
        setting = Settings(k)

        # setting = settings[k]

        tmp_worker = Worker(v, setting)
        with ThreadPool(processes = mp.cpu_count()) as p:
          p.map(tmp_worker, tmp_worker.idx)
          v = np.frombuffer(tmp_worker.buff, dtype = np.float64).reshape(*tmp_worker.buffshape)
        del tmp_worker
        s = v.shape
        v = v.reshape(-1, setting.nNodes, setting.deltas // 2 - 1)
        v = np.array([(i - i.min()) / (i.max() - i.min()) for i in v])
        v = v.reshape(-1, s[-1])
        setting.data = v.reshape(s)
#        loadedData[k] = v.reshape(s)
        settings[k] = setting
        pbar.update()
    return settings
class DataLoader(OrderedDict):
    def __init__(self, root = '', **kwargs):
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
     dataDir = root
     super(DataLoader, self).__init__(**kwargs)

     allowedExtensions = "pickle ".split()
     #TODO :  make aggregate dataclass -> how to deal with multiple samples
     # current work around is bad imo
     pattern = '((\d+-\d+-\d+T\d+:\d+:)?(\d+\.\d+))'
     if dataDir:
         # Warning: this only works in python 3.6+ due to how dictionaries retain order
         print("Extracting data...")
         files = []
         # walk root and find all the possible data files
         for root, dir, fileNames in os.walk(dataDir):
             for fileName in fileNames:
                 for extension in allowedExtensions:
                     if fileName.endswith(extension):
                         files.append(os.path.join(root, fileName))
                         break # prevent possible doubles
         print(f"Found {len(files)} of files")
         def tmp(x):
             """
             Look for the time in order to sort the data properly
             """
             x = x.split('/')[-1].split('_')[0]
             d = re.search(pattern, x)
             if d:
                 return d.group()
             else:
                 return ''
         files = sorted(files, key = lambda x: \
                                    tmp(x))

#         common = ''
#         for fs in zip(*files):
#             fs = set(fs)
#             if len(fs) == 1:
#                 common += str(list(fs)[0])
#

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
                 # temp = re.search('t=\d+\.[0-9]+', file).group

                 temp = file.split('/')[-3] # magnetization
                 root = os.path.join(*list(file.split('/'))[:-3])
                 # root = '/' + root
                 # print(root)
                 # print(root)
                 # root = os.path.join(*list(file.split('//'))[0:-3])
                 # print(root)
                 # deltas = re.search('deltas=\d+', file).group()
                 # deltas = re.search('\d+', deltas).group()
                 # look for pulse
                 pulse = re.search("\{.*\}", file).group()
                 structure = [root, temp]
                 if pulse == '{}':
                     structure += ['control']
                 else:
                     # there is a bug that there is a white space in my data;
                     structure += pulse[1:-1].replace(" ", "").split(':')[::-1]
                 # tmp  = loadPickle(file)
                 self.update(addData(data, file, structure))
             # attempt to load with certain properties, if not found gracdefully exit
             # this is the case for every pickle file other than the ones with props above
             except:
                 continue
         print('Done')



# class DataLoader:
    # def __init__(self):
        # self.data = {}

    # def load_data(self, root, types = ''):
    #

def flatLoader(root):
    """
    Walks root directory and returns flatten set of
    data filenames sorted based on date time in filename
    """

    # pattern is backwards compatible
    data, pattern = set(), '((\d+-\d+-\d+T\d+:\d+:)?(\d+\.\d+))'
    for root, dirs, files in os.walk(root):
        # will fail on files not containing date
        try:
            files = sorted(files, key = lambda x:\
                       re.search(pattern, x).group())
            for file in files:
                print(f'Flatloader... Loading: {file}')
                if file.endswith('pickle') and not file in data:
                    data.add(os.path.join(root, file))
        except:
            pass
    return data

# begin the uggliness
from Utils import misc, stats
import multiprocessing as mp
from tqdm import tqdm
import pyprind as pr
class Worker:
    def __init__(self , datadict, setting):
        self.setting = setting

        # ctime is bugged on linux(?)
        self.filenames = sorted(\
                           misc.flattenDict(datadict), \
                           key = lambda x: x.split('/')[-1].split('_')[0], \
                           )

        self.deltas = setting.deltas // 2 - 1
        numberOfNudges = len(setting.pulseSizes)

        expectedShape = (\
                         setting.nNodes * numberOfNudges  + 1,\
                         setting.nTrials,\
                         numberOfNudges + 1,\
                         )


#        DELTAS_, EXTRA = divmod(setting.deltas, 2) # extract relevant time data
#        DELTAS  = DELTAS_ - 1

        # linearize this shape
        # nudges and control hence + 1
        bufferShape = (\
                       numberOfNudges + 1, \
                       len(datadict),\
                       setting.nTrials,\
                       setting.nNodes,\
                       self.deltas, \
                       )
        # double raw array
        from multiprocessing import sharedctypes
        self.buff = sharedctypes.RawArray('d', int(np.prod(bufferShape)))
        self.buffshape     = bufferShape
        self.expectedShape = expectedShape
        """
        The data is filled by control (1) -> nudges system per nudge size
        for example for a system of 10 nodes with 2 conditions we would have
        (1 + 10 + 10) files for a full set
        """

        fullset = numberOfNudges * setting.nNodes + 1
        # if not completed; extracted only fullsets
        # i.e. when we have 20 out of full 21 we only want to have less
        # trials extracted otherwise we introduce artefacts when plotting data
        a, b = divmod(len(self.filenames), fullset)
        c = 1 if b else 0
        N = fullset * (a - c)
        # self.pbar = tqdm(total = N)
        self.pbar = pr.ProgBar(N)
        #    print(a, b, c, COND * NODES + 1 )
        self.idx = list(range(N))

    def __call__(self, fidx):
        """
        Temporary worker to load data files
        This function does the actual processing of the correct target values
        used in the analysis below
        """
        # shorthands
        fileName  = self.filenames[fidx]
        fileNames = self.filenames
        setting   = self.setting
        # do control
        data = np.frombuffer(self.buff).reshape(self.buffshape)
        node, temp, trial = np.unravel_index(fidx, self.expectedShape, order = 'F')
        # control data
        if '{}' in fileName:
            # load control; bias correct mi
            control = loadData(fileName)
            # panzeri-treves correction
            mi   = control.mi
            bias = stats.panzeriTrevesCorrection(control.px,\
                                                 control.conditional, \
                                                 setting.repeats)
            mi -= bias
            data[0, trial, temp]  = mi[:self.deltas, :].T
        # nudged data
        else:
            targetName = fileName.split('=')[-1].strip('.pickle') # extract relevant part
            # extract pulse and only get left part
            targetName = re.sub('{|}', '', targetName).split(':')[0]
            # get the idx of the node
            nodeNames = []
            for name, idx in setting.mapping.items():
                if name in targetName:
                    nodeNames.append(idx)
                    # print(idx, name, targetName)

            # load the corresponding dataset to the control
            controlidx = fidx - node
            assert '{}' in fileNames[controlidx]
            # load matching control
            control  =  loadData(fileNames[controlidx])
             # load nudge
            sample   = loadData(fileName)
            # impact = stats.KL(control.px, sample.px)
            impact   = stats.KL(sample.px, control.px, exclude = nodeNames)
            # don't use +1 as the nudge has no effect at zero
            redIm    = np.nansum(impact[-self.deltas:], axis = -1).T
            # TODO: check if this works with tuples (not sure)
            for name in nodeNames:
                data[(node - 1) // setting.nNodes + 1, trial, temp, name,  :] = redIm.squeeze().T
        self.pbar.update()

# TODO: careful
def loadData(root):
    files  = []
    for (root, dirs, file) in os.walk(root):
        if file.endswith('pickle'):
            files.append(file)
        return loadPickle(filename)
    if root.endswith('.pickle'):
        return loadPickle(root)

#def loadData(filename):
    # TODO: change
#    if filename.endswith('pickle'):
#        return loadPickle(filename)
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
    # safeguard?
    if not fileName.endswith('.pickle'):
        fileName += '.pickle'
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
import importlib
@dataclass
class Settings:
    """
    Reads json file and converts in a convenient python dataclass
    """
    # simulation parameters
    repeats       : int
    deltas        : int
    nSamples      : int
    step          : int
    burninSamples : int
    nNodes        : int
    nTrials       : int
    pulseSizes    : list

    nudgeType     : str

    # model properties
    updateType  : str
    # legacy support needed of these


    # defaults
    _graph        : dict  = field(init = False, repr = False, default_factory = dict)
    graph         : dict
    rmapping      : dict
    mapping       : dict
    model         : str   = 'Ising'
    _mapping      : dict  = field(init = False, repr = False, default_factory = dict)
    _rmapping     : dict  = field(init = False, repr = False, default_factory = dict)
    directory     : str   = field(init = False)  # "data directory"

    def __init__(self, data = None):
        """
        Input:
            :data: either dict or string. If dict it assigns keys as fields.
            If string it excpects a root folder which it will search for settings
            or emulate them
        """
        self._use_old_format_ = False
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
        # assume we wan't to load it from a file
        if isinstance(val, property):
            for root, subdirs, files in os.walk(self.directory):
                for file in files:
                    try:
                        graph  = loadData(os.path.join(root, file)).graph
                        if graph:
                            print("Graph found in data file!")
                            self._graph = nx.node_link_data(graph)
                            self._use_old_format_ = True
                            return 0
                    except AttributeError:
                        continue
            else:
                raise ValueError("Input graph not recognized")
        # if present in the data just load it
        elif isinstance(val, dict):
            self._graph = val
#            self._graph = nx.node_link_data(val)
        print('loading graph')

    def loadModel(self):
        """
        """
        # not dotted anymore need to look for a class name in the directory of
        # the models

        # look for model name inside the module files
        for file in os.listdir('Models/'):
            if file.endswith('pyx'):
                tmp = importlib.import_module(f'Models.{file.split(".")[0]}')
                if getattr(tmp, self.model, None):
                    # returned init model
                    # backward compatibility
                    if isinstance(self.graph, nx.Graph) or isinstance(self.graph, nx.DiGraph):
                        return getattr(tmp, self.model)(graph = self.graph)
                    # new method
                    else:
                        print('loading model', self._use_old_format_)
                        g =  nx.node_link_graph(self.graph)
                        if self._use_old_format_:
                            g.__version__ = 1.0
                        return getattr(tmp, self.model)(graph = g)
        return None
#        m = self.model.split('.')
#        mt = '.'.join(i for i in m[:-1]) # remove extension
#        return getattr(importlib.import_module(mt), m[-1])(self.graph)


    # add simple check instead of sideways?
    @property
    def mapping(self):
        return self._mapping
    @mapping.setter
    def mapping(self, val):
        """
        Extracts mapping either from dict or attempts to load the model
        """
        if isinstance(val, property):
            model          = self.loadModel()
            self._mapping  = model.mapping
        elif isinstance(val, dict):
            self._mapping = val
    @property
    def rmapping(self):
        return self._rmapping
    @rmapping.setter
    def rmapping(self, val):
        """
        Extracts rmapping either from dict or attempts to load the model
        """
        if isinstance(val, property):
            # implicitly works due to how the vars are loaded
            # mapping is loaded first and thus only needs constructor
            # self._rmapping = {j: i for i, j in self.mapping.items()}
            model = self.loadModel()
            self._rmapping  = model.rmapping

        elif isinstance(val, dict):
            self._rmapping = val


    def addItems(self, data):
        """
        Helper function for read; gives warning for unintended items
        """
        for field in fields(self):
            fetched = data.get(field.name, field.default)
            # ignore missing types and privates
            if not isinstance(fetched, _MISSING_TYPE) and not field.name.startswith('_'):
                self.__setattr__(field.name, fetched)
            elif not field.name.startswith('_'):
                print(f"WARNING: {field.name.replace('_','')} not found in input")

    def read(self, targetDirectory):
        """
        Reads json settings from target directory
        with minor backward compatibility
        """
        print(f"Reading settings in {targetDirectory}: ", end = '')
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

    def save(self, targetDirectory = None):
        targetDirectory = self.directory if targetDirectory is None else targetDirectory
        print('Saving settings')
        s = {k.replace('_', '') : v for k, v in self.__dict__.items()}
        print(s)
        with open(os.path.join(targetDirectory, 'settings.json'), 'w') as f:
            json.dump(\
                      s, f,\
                      default = lambda x : float(x), \
                      indent  = 4)
    def __repr__(self):
        """
        Print all settings
        """
        banner = '-' * 32
        top    = f'\n{banner} Simulation Settings {banner}'

        s = top
        for k, v in self.__dict__.items():
            if not k.startswith('_'):
                s += f'\n{k:<15} = {v}' # basic alignment[magic number]
        s += '\n'
        s += '-' * len(top)
        return s

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
#    graph       : object
    # TODO: add these?
    # temperature : int
    # pulse       : dict

if __name__ == '__main__':
    dp = '/home/casper/projects/information_impact/Data/'
    psycho   = '1548025318.5751357'
    settings = Settings(os.path.join(dataPath, psycho))
    print(settings)
