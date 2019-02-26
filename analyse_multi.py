import matplotlib.pyplot as plt, numpy as np, scipy, multiprocessing as mp, os, re
try:
    if __IPYTHON__:
        from tqdm import tqdm_notebook as tqdm
except NameError:
    from tqdm import tqdm
from Utils import IO, stats, misc

"""
- work from a folder directory
"""

root =  '/run/media/casper/test/1550482875.0001953/'

data     = IO.DataLoader(root) # extracts data folders
settings = {key : IO.Settings(os.path.join(root, key)) for key in data} # load corresponding settings


# begin the uggliness
class Worker:
    def __init__(self , datadict, setting):
        self.setting = setting
        self.filenames = sorted(\
                           misc.flattenDict(datadict), \
                           key = lambda x: os.path.getctime(x), \
                           )

        self.deltas = setting.deltas // 2
        numberOfNudges = len(setting.pulseSizes)

        expectedShape = (\
                         setting.nNodes * numberOfNudges  + 1,\
                         setting.nTrials,\
                         numberOfNudges + 1,\
                         )
        
        
        DELTAS_, EXTRA = divmod(setting.deltas, 2) # extract relevant time data
        DELTAS  = DELTAS_ - 1
        # linearize this shape
        bufferShape = (\
                       numberOfNudges + 1, \
                       len(datadict),\
                       setting.nTrials,\
                       setting.nNodes,\
                       DELTAS, \
                       )
        # double raw array
        self.buff = mp.RawArray('d', int(np.prod(bufferShape)))
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
        self.pbar = tqdm(total = N)
        #    print(a, b, c, COND * NODES + 1 )
        self.idx = list(range(N))
    def __call__(self, fidx):
        """
        Temporary worker to load data files
        This function does the actual processing of the correct target values
        used in the analysis below
        """

        fileName = self.filenames[fidx]
        setting = self.setting
        # do control
        data = np.frombuffer(self.buff).reshape(self.buffshape)
        node, temp, trial = np.unravel_index(fidx, self.expectedShape, order = 'F')
        # load settings
        t     = fileName.split('/')[-4]
        setting = settings[t] # load from global ; bad


        DELTAS_, EXTRA = divmod(setting.deltas, 2) # extract relevant time data
        DELTAS  = DELTAS_ - 1
        # TODO: check  this
        # only use half of the data

        # control data
        if '{}' in fileName:
            # load control; bias correct mi
            control = IO.loadData(fileName)
            dir     = fileName.split('/')[-4]
            # panzeri-treves correction
            mi   = control.mi
            bias = stats.panzeriTrevesCorrection(control.px,\
                                                 control.conditional, \
                                                 setting.repeat)
            mi -= bias
            data[0, trial, temp]  = mi[:DELTAS - EXTRA, :].T
        # nudged data
        else:
            targetName = fileName.split('_')[-1] # extract relevant part
            # get the idx of the node
            jdx = [setting.mapping[int(j)] if j.isdigit() else setting.mapping[j]\
                                 for key in setting.mapping\
                                 for j in re.findall(str(key), re.sub(':(.*?)\}', '', targetName))]
            jdx = jdx[0]
            # load the corresponding dataset to the control
            useThis = fidx - node
            # load matching control
            control = IO.loadData(self.filenames[useThis])
             # load nudge
            sample  = IO.loadData(fileName)
            impact  = stats.KL(control.px, sample.px)
            # don't use +1 as the nudge has no effect at zero
            redIm = np.nanmean(impact[DELTAS + EXTRA + 2:], axis = -1).T
            # TODO: check if this works with tuples (not sure)
            condition = (node - 1) // setting.nNodes + 1
            data[condition, trial, temp, jdx] = redIm.squeeze().T
        self.pbar.update(1)

loadedData = {}
for key in data:
    setting = settings[key]
    processes = mp.cpu_count()
    worker = Worker(data[key], setting)
    from multiprocessing.pool import ThreadPool
    
    with ThreadPool(processes = processes) as p:
        p.map(worker, worker.idx)
    loadedData[key] = np.frombuffer(worker.buff, dtype = np.float64).reshape(*worker.buffshape)
