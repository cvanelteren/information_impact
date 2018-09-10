import IO, plotting as plotz, os
from numpy import *
from matplotlib.pyplot import *
import networkx as nx
import networkx as nx
from tqdm import tqdm_notebook as tqdm
#%load_ext autoreload
#%autoreload 2
# %%
close('all')


res = {}
d = f'{os.getcwd()}/Data/'
def newest(path):
    files = os.listdir(path)
    paths = [os.path.join(path, basename) for basename in files]
    return sorted(paths, key=os.path.getctime)
d = newest(d)[-1]
#d = f'{os.getcwd()}/Data/data_single_chain'
# d = newest(d)
print(d)

# d = f'{os.getcwd()}'
colors = cm.tab20(arange(12))
res = {}
for file in os.listdir(d):
    if file.endswith('.pickle') and not 'mags' in file:
        print(file)
        tmp = file.split('_')
        pulse = tmp[-2].split('=')[-1]
        temp  = round(float(tmp[-4].split('=')[-1]), 3)
        res[temp] = res.get(temp, ()) + ((pulse, IO.loadPickle(f'{d}/{file}')), )

rres = {}
for i, j in res.items():
    rres[i] = dict(j)
k = list(rres.keys())

print(k)

# %%

import scipy
from tqdm import tqdm
tmp = rres[k[-1]]

H = zeros((34, 2))
func = lambda x, a, b, c, d, e, f : a + b * exp(-c*x) #  + d * exp(-e *(x - f))
fr   = lambda x, a, b : func(x, *a) - b


#%matplotlib inline
tester = {}

for idx, (i, j) in enumerate(tqdm(tmp.items())):
    mi = j['mi']
    snapshots = j['snapshots']
    joint = j['joint']
    model = j['model']
    colors = cm.tab20(arange(model.nNodes))
#     if i == '{}':
    fig, ax = subplots()
    tester[i] = array([snapshots[ii] * jj for ii, jj in joint.items()])
    for jdx, MI in enumerate(mi.T):
        x = arange(len(MI))
        a, b = scipy.optimize.curve_fit(func, x, MI, maxfev = 1000000)
#        print(a)
        l =  a[0] + .01
        if i == '{}':
            print(model.rmapping[jdx], a[0], l)
#        l = .001
    #         l = .2 * max(MI)
    #             l = 2/np.e
        r = scipy.optimize.root(fr, 0, args = (a, l))
        rot = r.x
        if rot < 0:
            rot = 0
        xx = linspace(0, 300, 1000)
        if i == '{}':
            H[jdx, 0] = 0 if max(MI) <= l else rot
        else:
            H[jdx, 1] += rot / model.nNodes
        ax.plot(xx, func(xx, *a), '--', color = colors[jdx], alpha = 1)
        ax.scatter(x, MI, color = colors[jdx], label = model.rmapping[jdx],\
                   alpha = 1)
        ax.scatter(rot, func(rot, *a), \
                       color = colors[jdx], marker = 's', s = 100)
        ax.set_xlim(-.2, 3)
        ax.set_ylim(-.2, 1)
        ax.set_title(i)
        ax.legend(bbox_to_anchor = (1., .5))
#    if i != '{}':
#        close()
# %%
fig, ax = subplots()
[ax.scatter(*H[idx, :], color = colors[idx]) for idx in range(model.nNodes)]

ii = min(H[:,0]), max(H[:, 0])
ax.plot(ii,ii , '--k', alpha = .2)
fig, ax = subplots()
w = None
w = 'weight'
degs = dict(nx.degree(model.graph, weight = w))
#degs = dict(nx.betweenness_centrality(model.graph, normalized = 0))

degs = dict(nx.closeness_centrality(model.graph))
#degs = dict(nx.eigenvector_centrality(model.graph))

tmp1 = degs.copy()
m, mm = min(degs.values()), max(degs.values())
for i, j in tmp1.items():
    tmp1[i] = (j - m) / (mm - m)
degs = tmp1
for node, deg in degs.items():
    idx = model.mapping[node]
    ax.scatter(deg, H[idx, 0], color = colors[idx], label = node)
#     ax.scatter(*H[idx, :], color = colors[idx])
#ax.set_yscale('log')
setp(ax, **dict(xlabel = 'degree', ylabel = 'abs IDT'))
ax.legend(bbox_to_anchor = (1.01, 1))

show()


# %%
#fig, ax = subplots()
#degs = dict ( nx.degree(model.graph))
#for i, j in degs.items():
#    idx = model.mapping[i]
#    ax.scatter(j, mi[0, idx])
#setp(ax, **dict(xlabel = 'degree', ylabel = 'entropy'))
## %%
pxs = {}
for i, j in tmp.items():
    shape = list(joint.values())[0].shape
    px = zeros((shape))
    joint = j['joint']
    snaps = j['snapshots']

    for ii, jj in joint.items():
        px += snaps[ii] * jj
    pxs[i] = px
# %%
hd = lambda x, y : sqrt(((sqrt(x) - sqrt(y))**2).sum(-1))/sqrt(2)
idx = -1
control = pxs['{}'][idx, ...]
hs = {}
for i, j in pxs.items():
    if i != '{}':
        title = i.split(':')[0].split('{')[1].split("'")[1]
        
        h = hd(control, j[idx, ...])
        hs[title] = h.mean()
fig, ax = subplots(figsize = (10, 10))
ax.bar(list(hs.keys()), list(hs.values()))
ax.set_ylabel('Hellinger distance')
# %%
fig, ax = subplots()
for node in model.graph.nodes():
    x = hs[node]    
    y = H[model.mapping[node], 0]
    ax.scatter(x, y, color = colors[model.mapping[node]], label = node)
setp(ax, **dict(xlabel = 'impact', ylabel = 'idt'))
ax.legend(bbox_to_anchor = (1.01, 1))
show()
