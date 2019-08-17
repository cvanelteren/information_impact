"""
This scripts controls the simulation and claims seperately a number of compute units.
It exists to circumvent the 48 hours claim per job.

Structure is as follows

control.py: main control script. It stores temporary pickle files that the other process have to delete.
general settings have to be setup here

"""
import subprocess, datetime, numpy as np, networkx as nx
from Models import FastIsing



epeats        = int(1e5)
deltas        = 30
step          = int(1e3)
nSamples      = int(1e4)
burninSamples = 0
pulseSizes    = np.linspace(.1, 1, 9).tolist()
# pulseSizes  = np.arange(0.5, 5, .5) # np.linspace(0.5, 5, 5).tolist() # [0.5, np.inf] #, -np.inf]# , .8, .7]

nTrials       = 20
magSide       = ''
updateType    = '0.25'
CHECK         = [0.8] # , .5, .2] # if real else [.9]  # match magnetiztion at 80 percent of max
nudgeType     = 'constant'
tempres       = 10 #100


genDataFile = lambda x : f'dataset{idx}''
for idx, graph in enumerate(graphs):
    start = datetime.datetime.now()
    targetDirectory = os.path.join(rootDirectory, f'{start.isoformat()}')
    now = datetime.datetime.now().isoformat()
    # if multiple graphs are tested; group them together
    if not os.path.exists(rootDirectory):
        os.mkdir(rootDirectory)
    os.mkdir(targetDirectory)

    # graph = nx.barabasi_albert_graph(10, 3)
    modelSettings = dict(\
                         graph       = graph,\
                         temperature = 0,\
                         updateType  = updateType,\
                         magSide     = magSide,\
                         nudgeType   = nudgeType)
    model = FastIsing.Ising(**modelSettings)
    updateType = model.updateType

    settings = dict(
                    repeats       = repeats,\
                    deltas        = deltas,\
                    nSamples      = nSamples,\
                    step          = step,\
                    burninSamples = burninSamples,\
                    pulseSizes    = pulseSizes,\
                    updateType    = updateType,\
                    nNodes        = graph.number_of_nodes(),\
                    nTrials       = nTrials,\
                    # this is added
                    graph         = nx.readwrite.json_graph.node_link_data(graph),\
                    mapping       = model.mapping,\
                    rmapping      = model.rmapping,\
                    model         = type(model).__name__,\
                    directory     = targetDirectory,\
                    nudgeType     = nudgeType,\
                    )
    runFile = genDataFile(idx)
    IO.savePickle(runFile, settings)

    cmd = f"sbatch control_run.sh {runFile}".split()
    subprocess.call(cmd)
