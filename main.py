import numpy as np, networkx as nx, os, datetime, itertools, \
copy, time
from Utils import IO

from subprocess import call, Popen
# compile
call('python compile.py build_ext --inplace'.split())
from Models import FastIsing
"""
Things to do:
I need a way for the simulations to be self-contained. Creating a
 specific files will mess with possible future work. Preferably, we want
 some form of setting file to take care of anything specific. The models should therefore
 include the relevant settings and initiate the files.

Additionally, there is a need to setupt the nudges little more proper


- Magnetization [x]
- Add control to sims []
"""
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


settings = dict(\
    model         = FastIsing.Ising,\
    repeats       = int(1e4),\
    deltas        = 30,\
    steps         = int(1e3),\
    nSamples      = int(1e4),\
    burninSamples = 0,\
    nTrials       = 50,\
    modelSettings = modelSettings,\
    tempres       = 100,\
    )

# control is over-counted
# check if the file is running on das4
nodename = os.uname().nodename
if any([nodename in i for i in \
'fs4 node'.split()]):
    now = datetime.datetime.now().isoformat()
    rootDirectory = f'/var/scratch/cveltere/{now}'
    runCommand = 'sbatch single_run.sh'
else:
    rootDirectory = f'{os.getcwd()}/Data'
    runCommand = 'python3 single_run.py --file'


pulseSizes    = np.arange(0, 5, .5).tolist()
models = []

def createJob(model, settings, root = ''):
    tmp                       = os.path.join(root, 'data')

    os.makedirs(tmp, exist_ok = 1)

    mag                       = settings.get('ratio')[0]
    trial                     = settings.get('trial')
    intervention              = settings.get('pulse')
    now                       = datetime.datetime.now().isoformat()
    fileName                  = f"trial={trial}_r={mag}_{intervention}.pickle"
    fileName                  = os.path.join(tmp, fileName)
    return fileName

# init models
for _ in range(10):
    g = nx.erdos_renyi_graph(10, np.random.uniform(.2, .8))
    m = settings.get('model')(graph = g, \
                            **settings.get('modelSettings'), \
                            equilibrium = equilibrium)
    # m.equilibriate(magRatios, eqSettings)
    combinations = itertools.product(\
        m.matched['ratios'].items(),\
        pulseSizes, \
        range(settings.get('nTrials'))
    )
    settings['graph'] = g
    settings['model'] = m
    # setup filepaths
    now = datetime.datetime.now().isoformat()
    simulationRoot = os.path.join(\
                    rootDirectory, now)
    print(f"Making {simulationRoot}")
    os.makedirs(simulationRoot, exist_ok = 1)

    settings['equilibrium'] = m.matched
    IO.savePickle(os.path.join(simulationRoot, 'settings'),\
                settings)

    for (ratio, pulse, trial) in combinations:

        # tmp added properties
        settings['trial'] = trial
        settings['ratio'] = ratio

        if pulse:
            for node in m.graph.nodes():
                tmp = copy.deepcopy(m)
                tmp.t = ratio[1]
                intervention = {node : pulse}
                tmp.nudges = intervention

                settings['pulse'] = intervention
                settings['model'] = tmp

                fn = createJob(tmp, settings, simulationRoot).replace(' ', '')
                IO.savePickle(fn, copy.deepcopy(settings))
                Popen([*runCommand.split(), fn])
                time.sleep(.1)
                # call(f'sbatch single_run.sh {fn}'.split())
                # print(fn)
        else:
            tmp = copy.deepcopy(m)
            tmp.nudges = {}
            tmp.t = ratio[1]

            settings['pulse'] = {}
            settings['model'] = tmp

            fn = createJob(tmp, settings, simulationRoot).replace(' ', '')
            IO.savePickle(fn, copy.deepcopy(settings))
            Popen([*runCommand.split(), fn])
            time.sleep(.1)
