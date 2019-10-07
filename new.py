import numpy as np, networkx as nx, \
        os, datetime, itertools, \
        copy, time, argparse

from Toolbox import infcy
from Utils import IO


from subprocess import call, Popen
from Models import FastIsing

modelSettings = dict(\
    magSide       = 'neg',\
    updateType    = '0.25',\
    nudgeType     = 'constant',\
    )

equilibrium   =  (\
    np.array([.8, .7, .6]),\
    dict(\
        n = 100, \
        temperatures = np.logspace(\
                        -3, np.log10(10), 10)\
                )
        )


pulseSizes    = np.linspace(.5, 5, 9).tolist()

settings = dict(\
    model         = FastIsing.Ising,\
    repeats       = int(1e4),\
    deltas        = 30,\
    steps         = int(1e3),\
    nSamples      = int(1e4),\
    burninSamples = 0,\
    nTrials       = 30,\
    modelSettings = modelSettings,\
    tempres       = 100,\
    pulseSizes    = pulseSizes\
    )

parser = argparse.ArgumentParser()
parser.add_argument('--file', default = None, type = str)
parser.add_argument('--id')
combinations = None
# deadline is 3600 * 48
THRESHOLD = time.time() + 3600 * 46
# THRESHOLD = time.time() + 10
def createJob(model, settings, root = ''):
    tmp                       = os.path.join(root, 'data')

    os.makedirs(tmp, exist_ok = 1)

    mag                       = settings.get('ratio')[0]
    trial                     = settings.get('trial')
    intervention              = settings.get('pulse')
    now                       = datetime.datetime.now().isoformat()
    fileName                  = f"trial={trial}_r={mag}_{intervention}.pickle"
    fileName                  = os.path.join(tmp, fileName)
    fileName = fileName.replace(' ', '')
    return fileName

def checkTime():
    """
    Save state on exit
    """
    import inspect, sys

    global PID
    if time.time() > THRESHOLD:

        globs = {}
        for k, v in globals().copy().items():
            if not inspect.ismodule(v) and not isinstance(v, argparse.ArgumentParser):
                globs[k] = v
        if PID is None:
            PID = 1234
        simFile = f'sim-{PID}'
        IO.savePickle(simFile, globs)
        sys.exit()

def runJob(model, settings, simulationRoot):
    """
    Run the job and stops process if taking too long"
    """
    global rootDirectory
    if settings.get('pulse'):
        trial = settings.get('trial')
        mag   = settings.get('ratio')[0]

        control = f'data/trial={trial}_r={mag}_{{}}.pickle'
        control = os.path.join(simulationRoot, control)

        snapshots = {}
        # try to load the snapshots
        # redundant check if run on separate process
        while not snapshots:
            try:
                snapshots = IO.loadPickle(control).snapshots
            except:
                time.sleep(1)

    else:
        snaps = {}
        for k in 'nSamples burninSamples steps'.split():
            snaps[k] = settings.get(k)
        snapshots = infcy.getSnapShots(model, **snaps)

    conditional, px, mi = infcy.runMC(model, snapshots, deltas, repeats)
    store       = dict(\
        mi          = mi,\
        conditional = conditional,\
        px          = px,\
        snapshots   = snapshots)

    # empty vector to safe disk space
    if settings.get('pulse'):
        for i in 'mi conditional snapshots'.split():
            store[i] = []
    fn = createJob(model, settings, simulationRoot)
    sr = IO.SimulationResult(**store)
    IO.savePickle(fn, sr, verbose = 1)
    checkTime()

 # init models
if __name__ == "__main__":
    M    = settings.get('model')
    args = parser.parse_args()
    file, PID = args.file, args.id
    print(file, PID)
    # this should only be run once per call
    if not file:
        g = nx.erdos_renyi_graph(10, np.random.uniform(0, 1))
        m = M(graph = g, \
            **settings.get('modelSettings'), \
            equilibrium = equilibrium)

        pulses = list(\
                itertools.product(pulseSizes, \
                                ['', *list(\
                                m.graph.nodes())\
                                ]))

        combinations = itertools.product(\
                m.matched['ratios'].items(),\
                pulses, \
                range(settings.get('nTrials'))
                )

        settings['graph'] = g
        settings['model'] = M
        
        # setup filepaths
        nodename = os.uname().nodename
        now = datetime.datetime.now().isoformat()
        if any([i in nodename for i in 'fs4 node'.split()]):
            rootDirectory = f'/var/scratch/cveltere/{now}'
        else:
            rootDirectory = f'{os.getcwd()}/Data/{now}'

        simulationRoot = os.path.join(\
                    rootDirectory, now)

        print(f"Making {simulationRoot}")
        os.makedirs(simulationRoot, exist_ok = 1)

        settings['equilibrium'] = m.matched
        # make settings
        IO.savePickle(os.path.join(simulationRoot, 'settings'),\
                settings)
    # resume process
    else:
        print('Resuming process')
        for k, v in IO.loadPickle(file).items():
            if k != 'file':
                globals()[k] = v
        os.remove(f'{file}')



    deltas, repeats = [\
            settings.get(k) \
            for k in 'deltas repeats'.split()]

    ccombinations = copy.deepcopy(combinations)
    combinations = list(combinations)
    for (ratio, (pulse, node), trial) in ccombinations:
        # tmp added properties
        settings['trial'] = trial
        settings['ratio'] = ratio
        if node:
            tmp          = copy.deepcopy(m)
            tmp.t        = ratio[1]
            intervention = {node : pulse}
            tmp.nudges   = intervention

            settings['pulse'] = intervention
            settings['model'] = tmp
            runJob(tmp, settings, simulationRoot)

        else:
            tmp        = copy.deepcopy(m)
            tmp.nudges = {}
            tmp.t = ratio[1]

            settings['pulse'] = {}
            settings['model'] = tmp
            runJob(tmp, settings, simulationRoot)
        combinations.pop(0)
