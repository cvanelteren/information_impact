
import matplotlib as mpl, matplotlib.pyplot as plt
from Models import  fastIsing, potts
import networkx as nx, numpy as np
from Utils import IO
import time
n = 500


# g = nx.grid_2d_graph(n, n)
# g = nx.star_graph(10)


 #        graphs += [nx.barabasi_albert_graph(n, i) for i in linspace(2, n - 1, 3, dtype = int)]
# #        graphs += [nx.barabasi_albert_graph(n, i) for i in linspace(2, n - 1, 3, dtype = int)]
dataDir = 'Psycho' # relative path careful
df    = IO.readCSV(f'{dataDir}/Graph_min1_1.csv', header = 0, index_col = 0)
h     = IO.readCSV(f'{dataDir}/External_min1_1.csv', header = 0, index_col = 0)
g   = nx.from_pandas_adjacency(df)
attr = {}
for node, row in h.iterrows():
    attr[node] = dict(H = row['externalField'], nudges = 0)
nx.set_node_attributes(g, attr)
#
#attr = {}
#for node, row in h.iterrows():
#    attr[node] = dict(H = row['externalField'], nudges = 0)
#nx.set_node_attributes(g, attr)



#g.add_edge(0,0)
#g =  nx.soft_random_geometric_graph(20, .2)
#g = nx.grid_2d_graph(10, 10)
#g = nx.path_graph(3, nx.DiGraph()).
#w = nx.utils.powerlaw_sequence(20, exponent = 1.2)

#while True:
#    g = nx.erdos_renyi_graph(30, .05)
#    g = sorted(nx.connected_component_subgraphs(g), key = len)[-1]
#    if len(g) == 20:
#        break
    


dataDir = 'Graphs' # relative path careful
df    = IO.readCSV(f'{dataDir}/Graph_min1_1.csv', header = 0, index_col = 0)
h     = IO.readCSV(f'{dataDir}/External_min1_1.csv', header = 0, index_col = 0)
g   = nx.from_pandas_adjacency(df)
attr = {}
for node, row in h.iterrows():
    attr[node] = dict(H = row['externalField'], nudges = 0)
nx.set_node_attributes(g, attr)    
#g = nx.florentine_families_graph()

#g = nx.grid_2d_graph(n, n)
#from Models.percolation import Percolation
#n
#m = Percolation(graph = g, p = .55, updateType = 'async')
#m.reset()

#tmp = [j for i, j in m.mapping.items() if i == str((n//2, n//2))][0]
#b = np.zeros(m.nNodes, dtype = int)
#b[tmp] = 1
#m.states = b
#print(m.simulate(500))

#m.reset()
#a = m.simulate(1000)
#fig, ax = plt.subplots()
#ax.imshow(a.mean(0).reshape(n, n), aspect = 'auto')


n = 150
#g = nx.star_graph(5)
#plt.hist(w)
#g = nx.expected_degree_graph(w)
#for i, j in g.edges():
#    g[i][j]['weight'] = np.random.rand()  * 2 - 1
    
#g = nx.erdos_renyi_graph(10, .2)
#g = nx.watts_strogatz_graph(10, 3, .7)
#g = nx.full_rary_tree(3, 10)
#g = nx.duplication_divergence_graph(10, .25)
#g = nx.erdos_renyi_graph(10, .3)
#g = nx.grid_2d_graph(10, 10)
#g = nx.complete_graph(10)

#g = nx.erdos_renyi_graph(5, .25)
#g = nx.florentine_families_graph()
fig, ax = plt.subplots()
nx.draw(g, pos = nx.circular_layout(g), ax = ax, with_labels = 1)
fig, ax = plt.subplots();

deg = list(dict(g.degree()).values())
ax.hist(deg, bins = 20)
fig.show()


# %%

m = fastIsing.Ising(graph = g, \
                    updateType = 'single', \
                    magSide = 'neg', \
                    nudgeType = 'constant',\
                    nudges = {})
#m = potts.Potts(graph = g, agentStates = [1, 2])
#assert 0 
temps = np.logspace(-3, np.log10(g.number_of_nodes()), 50)
temps = np.linspace(0, 10, 50)
samps = [m.matchMagnetization(temps, 100) for i in range(10)]


from scipy import ndimage
samps = ndimage.gaussian_filter1d(samps,1, axis = 1)    
# %%
tsamps = np.array(samps)
tsamps = tsamps.mean(0)
mag, sus = tsamps
# %% 
import scipy
fig, ax = plt.subplots()
ax.scatter(temps, mag, color = 'orange', label = 'magnetization')

sig = lambda x, a, b, c, d:  c / (1 + np.exp(a *(x - b))) + d
coeffs, _ = scipy.optimize.curve_fit(sig, temps, mag, bounds = (0, np.inf), maxfev = 10000)
x0 = scipy.optimize.fmin(lambda x : abs(sig(x, *coeffs) - .8* mag.max()), .5)[0]


#idx = np.nanargmax(sus) 
#idx = np.argmin(abs(mag - .9 * mag.max()))
tax = ax.twinx()
tax.scatter(temps, sus, label = 'susceptibility')
ax.legend(); tax.legend(loc = 'lower right')
#ax.plot(temps[idx], mag[idx], 'r.')
tx = np.linspace(0, 10)
ax.plot(tx, sig(tx, *coeffs))
ax.axvline(x0, color = 'red')
ax.set(xlim = (0, 10))

print(x0)
fig.show()
m.t = x0

fig, ax = plt.subplots(); ax.scatter(temps, sus)

# %%
#assert 0 
#m.t = 1
#print(m.states.base)
from Toolbox import infcy
def entropy(p):
    return - (np.log2(p)*p).sum()

from itertools import product
#N = 10
#M = 100
#tmp = np.zeros((M, 2))

#snaps = list(product(m.agentStates, repeat = N))
#for i in range(M):
#    g = nx.erdos_renyi_graph(N, .2)
#    m = fastIsing.Ising(graph = g, \
#                    updateType = 'async', \
#                    magSide = '', \
#                    nudgeType = 'constant',\
#                    nudges = {}, t = np.random.rand() * 10)
#    snapshots    = infcy.getSnapShots(m, nSamples = int(1e3), steps = int(1e3),  nThreads = -1)
#
#
#    e = np.zeros(len(snaps))
#    for idx, s in enumerate(snaps):
#        m.states = np.asarray(s)
#        e[idx] = np.exp(-m.beta * m.hammy() / 2)
#    e /= e.sum()
#    print(e)
#    tmp[i, 1] = entropy(e)
#    tmp[i, 0] = entropy(np.fromiter(snapshots.values(), dtype = float))
#print(tmp)
#print(all(tmp[:, 0] < tmp[:, 1]))
#assert 0
# %%
colors = plt.cm.tab20(np.arange(m.nNodes))

#m = potts.Potts(graph = g, temperature = 0, \
#                 updateType = 'async', \
#                 agentStates = [1, 2, 3], \
#                 memorySize = 0, delta = 0.5)

## print(m.memorySize, m.memory.base)
temps = np.linspace(0, 10, 300)
#temps = np.logspace(-10, 1, 300)
##
N = 100
##print(m.memory.base.shape)
# a, b = m.matchMagnetization(temps, N)

from Toolbox import infcy
deltas = 50
start = time.time()
m.reset()
snapshots    = infcy.getSnapShots(m, nSamples = int(1e3), steps = int(1e3),  nThreads = -1)
repeats = int(1e4)
#assert False
                
#m.magSide = ''
conditional, px, mi = infcy.runMC(m, snapshots, deltas, repeats)
#

assert len(conditional) == len(snapshots)

from Utils.stats import KL, hellingerDistance, JS

# %%
NUDGE = .1 # np.inf
#NUDGE = .35
out = np.zeros((m.nNodes, deltas))
for node, idx in m.mapping.items():
    m.nudges = {node : NUDGE}
    c, p, n = infcy.runMC(m, snapshots, deltas, repeats)
    out[idx, :] = KL(px, p).sum(-1)
print(time.time() - start)
# %%
plt.close('all')


fig, (ax, tax) = plt.subplots(1, 2); 
#[ax.plot(i, color = colors[idx], label = m.rmapping[idx]) for idx, i in enumerate(mi.T)]
#ax.set_xlim(0,Q 4)
ax.legend(bbox_to_anchor = (1.01, 1))
ax.set(xlabel = 'time[step]', ylabel ='$I(s_i^{t_0 + t} : S^{t_0})$')
#ax.set_xlim(0, idx)

from Utils.plotting import fit
import scipy
x = np.zeros((m.nNodes, 2))

func = lambda x, a, b, c, d, e, f, g: a + b * np.exp(-c * (x - d)) + e * np.exp(- f * (x))

params = dict(\
             bounds = (0, np.inf),\
#             jac = 'cs', \
             maxfev = 100000)
xx =  np.linspace(0, deltas//2)
for idx, i in enumerate([mi.T, out[:, deltas // 2 + 1:-1]]):
    coeffs, _ = fit(i, func, params = params)
    for cidx, c in enumerate(coeffs):
        print(cidx, c)
        x[cidx, idx], _ = scipy.integrate.quad(lambda x: func(x, *c) - c[0], 0, np.inf)
        if idx == 0:
            ax.plot(xx, func(xx, *c), linestyle = 'dashed', color = colors[cidx])
            ax.plot(mi[:, cidx], color = colors[cidx])
        elif idx == 1:
            tax.plot(xx, func(xx, *c), linestyle = 'dashed', color = colors[cidx])
            tax.plot(i[cidx], color = colors[cidx])

#lims = (-.3, 30)
#ax.set(xlim = lims)
#tax.set(xlim = lims)
fig, ax = plt.subplots()

#x = np.trapz(mi[:deltas // 2 - 1, :], axis = 0)
#y = np.trapz(out[:, deltas // 2:], axis = -1)
[ax.scatter(*xy, color = ci) for ci,  xy in zip(colors, x)]
#ax.set(xscale = 'log')
fig.show()


fig, ax = plt.subplots()
nx.draw(g, ax = ax, pos = nx.circular_layout(g), with_labels = 1)

fig, ax = plt.subplots()

degs = list(dict(g.degree()).values())
ax.hist(degs, bins = 10)

# %%
#g = nx.duplication_divergence_graph(20, .2)
print(nx.is_connected(g))
fig, ax = plt.subplots()
nx.draw(g, ax = ax, pos = nx.circular_layout(g), with_labels = 1)


fig, ax = plt.subplots()

degs = list(dict(g.degree()).values())
ax.hist(degs, bins = 10)
plt.show()
