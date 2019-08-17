"""
This scripts controls the simulation and claims seperately a number of compute units.
It exists to circumvent the 48 hours claim per job.

Structure is as follows

control.py: main control script. It stores temporary pickle files that the other process have to delete.
general settings have to be setup here

"""
import subprocess, datetime, numpy as np, networkx as nx, os
from Models import FastIsing
from Utils import IO



repeats        = int(1e5)
deltas        = 30
step          = int(1e3)
nSamples      = int(1e4)
burninSamples = 0
pulseSizes    = np.linspace(.1, 1, 9).tolist()
# pulseSizes  = np.arange(0.5, 5, .5) # np.linspace(0.5, 5, 5).tolist() # [0.5, np.inf] #, -np.inf]# , .8, .7]

nTrials       = 20
magSide       = ''
updateType    = '0.25'
CHECK         = [.5, .2] # if real else [.9]  # match magnetiztion at 80 percent of max
nudgeType     = 'constant'
tempres       = 100


genDataFile = lambda x : f'dataset{idx}'


graphs = []
N  = 10
loadGraph = ''
if not loadGraph:
    for i in range(10):

        r = np.random.rand() * (1 - .2) + .2
        # g = nx.barabasi_albert_graph(N, 2)
        # g = nx.erdos_renyi_graph(N, r)
        # g = nx.duplication_divergence_graph(N, r)
        # graphs.append(g)
else:
    print('running craph graph')
    graph = IO.loadPickle(loadGraph)
    graphs.append(graph)
   # w = nx.utils.powerlaw_sequence(N, 2)
   # g = nx.expected_degree_graph(w)
    # g = sorted(nx.connected_component_subgraphs(g), key = lambda x: len(x))[-1]

    #for i, j in g.edges():
    #    g[i][j]['weight'] = np.random.rand() * 2 - 1
#        graphs.append(g)

#    graphs[0].add_edge(0,0)
#    for j in np.int32(np.logspace(0, np.log10(N-1),  5)):
#       graphs.append(nx.barabasi_albert_graph(N, j))
dataDir = 'Graphs' # relative path careful
df    = IO.readCSV(f'{dataDir}/Graph_min1_1.csv', header = 0, index_col = 0)
h     = IO.readCSV(f'{dataDir}/External_min1_1.csv', header = 0, index_col = 0)
graph   = nx.from_pandas_adjacency(df)
attr = {}
for node, row in h.iterrows():
    attr[node] = dict(H = row['externalField'], nudges = 0)
nx.set_node_attributes(graph, attr)
graphs.append(graph)

if 'fs4' in os.uname().nodename or 'node' in os.uname().nodename:
    now = datetime.datetime.now().isoformat()
    rootDirectory = f'/var/scratch/cveltere/{now}' # data storage
else:
    rootDirectory = f'{os.getcwd()}/Data/'
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
                    graph         = graph,\
                    mapping       = model.mapping,\
                    rmapping      = model.rmapping,\
                    model         = type(model).__name__,\
                    directory     = targetDirectory,\
                    nudgeType     = nudgeType,\
                    CHECK = CHECK,\
                    tempres = tempres)
    runFile = genDataFile(idx)
    IO.savePickle(runFile, settings)

    cmd = f"sbatch control_slurm.sh {runFile}".split()
    subprocess.call(cmd)
