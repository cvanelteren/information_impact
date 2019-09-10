import numpy as np, networkx as nx, os, datetime
from Utils import IO

from Models import FastIsing
"""
Things to do:
I need a way for the simulations to be self-contained. Creating a
 specific files will mess with possible future work. Preferably, we want
 some form of setting file to take care of anything specific. The models should therefore
 include the relevant settings and initiate the files.

Additionally, there is a need to setupt the nudges little more proper

"""
modelSettings = dict(\
    magSide       = 'neg',\
    updateType    = '0.25',\
    nudgeType     = 'constant',\
    )

settings = dict(\
    model         = FastIsing.Ising,\
    repeats       = int(1e4),\
    deltas        = 30,\
    step          = int(1e3),\
    nSamples      = int(1e4),\
    burninSamples = 0,\
    nTrials       = 50,\
    modelSettings = modelSettings,\
    tempres       = 100,\
    )
import itertools
pulseSizes    = np.arange(0, 5, .5).tolist()
nodes         = {}

eqSettings =    dict(\
                n = 100, \
                temperatures = np.linspace(0, 10, 10)\
                )
magRatios = np.array([.8, .7, .6])

models = []

for _ in range(10):
    g = nx.erdos_renyi_graph(10, np.random.uniform(0.2, .8))
    m = settings.get('model')(graph = g, \
                            **settings.get('modelSettings'))
    m.equilibriate(magRatios, eqSettings)
    print(m.matched['ratios'])
    print(m.matched['ratios'].get(0.8))
    models.append(m)
# control is over-counted
interventions = [itertools.product(pulseSizes, m.graph.nodes()) for m in models]

# check if the file is running on das4
nodename = os.uname().nodename
if any([nodename in i for i in \
'fs4 node'.split()]):
    now = datetime.datetime.now().isoformat()
    rootDirectory = f'/var/scratch/cveltere/{now}'
else:
    rootDirectory = f'{os.getcwd()}/Data'








            # make file
            # dispatch job
