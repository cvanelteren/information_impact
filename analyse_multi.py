
import matplotlib.pyplot as plt, numpy as np, scipy, multiprocessing as mp, os, \
re, networkx as nx

from tqdm import tqdm
from Utils import IO, stats, misc, plotting as plotz
from functools import partial

"""
- work from a folder directory
"""
# standard stuff
#root =  '/run/media/casper/test/1550482875.0001953/'
root = 'Data/cveltere/2019-05-09T16:10:34.645885'
root = '/run/media/casper/fc7e7a2a-73e9-41fe-9020-f721489b1900/cveltere'
root = 'Data/2019-05-13T13:34:02.290439'
root = 'Data/1548025318.5751357'
root = 'Data/cveltere/test'
#root = '/run/media/casper/4fdab2ee-95ad-4fc5-8027-8d079de9d4f8/Data/1548025318'

data     = IO.DataLoader(root) # extracts data folders
settings = {key : IO.Settings(root) for key in data} # load corresponding settings

centralities = {
                    r'$c_i^{deg}$' : partial(nx.degree, weight = 'weight'), \
                    r'$c_i^{betw}$': partial(nx.betweenness_centrality, weight = 'weight'),\
                    r'$c_i^{ic}$'  : partial(nx.information_centrality, weight = 'weight'),\
                    r'$c_i^{ev}$'  : partial(nx.eigenvector_centrality, weight = 'weight'),\
            }


#centralities = {key : partial(value, weight = 'weight') for key, value in nx.__dict__.items() if '_centrality' in key}

figDir = '../thesis/figures/'
information_impact = '$\mu_i$'
causal_impact      = '$\gamma_i$'
# %%

# %% load data
# functionize this; part of IO?
loadedData = {}
settings   = {}


import multiprocessing as mp
from multiprocessing.pool import ThreadPool
processes = mp.cpu_count()
for k, v in data.items():
    tmp = os.path.join(root, k)
    settings[k] = IO.Settings(tmp)
    worker = IO.Worker(v, settings[k])
    with ThreadPool(processes = processes) as p:
        p.map(worker, worker.idx)
    loadedData[k] = np.frombuffer(worker.buff, dtype = np.float64).reshape(*worker.buffshape)    
del worker
# %% # %% normalize data
# fit functions
double = lambda x, a, b, c, d, e, f: a + b * np.exp(-c*(x)) + d * np.exp(- e * (x-f))
double_= lambda x, b, c, d, e, f: b * np.exp(-c*(x)) + d * np.exp(- e * (x - f ))
single = lambda x, a, b, c : a + b * np.exp(-c * x)
single_= lambda x, b, c : b * np.exp(-c * x)
special= lambda x, a, b, c, d: a  + b * np.exp(- (x)**c - d)

func        = double
p0          = np.ones((func.__code__.co_argcount - 1)); # p0[0] = 0
fitParam    = dict(maxfev = int(1e6), \
                   bounds = (0, np.inf), p0 = p0,\
                   jac = 'cs')


# uggly mp case
# note use of globals here!
def worker(sample):
    auc = np.zeros((len(sample), 2))
    coeffs, errors = plotz.fit(sample, func, params = fitParam)
    for nodei, c in enumerate(coeffs):
        tmp = 0
        F      = lambda x: func(x, *c) - c[0]
        tmp, _ = scipy.integrate.quad(F, 0, LIMIT)
        auc[nodei, 0] = tmp
        auc[nodei, 1] = errors[nodei]
    auc[auc < np.finfo(auc.dtype).eps ] = 0
    return auc


def RejectOutliers(data, threshold = 3):

    """
    Rejects nodal outliers based on :threshold: away from the mean based on the
    mahalanobis distance
    """
    from sklearn.covariance import MinCovDet
    clf = MinCovDet()
    clf.fit(data)
    distances = clf.mahalanobis(data)

    outliers  = np.where(distances >= threshold)[0]
    inliers   = np.where(distances < threshold)[0]
    return inliers, outliers

information_impact = r'$\mu_i$'
causal_impact      = r'$\gamma_i$'

from scipy import ndimage
zds = {} # rescaled data
aucs= {} # area under the curve
for key, vals in loadedData.items():
    zd = vals.copy()
    rescale = True
#    rescale = False
    if rescale:
        s  = zd.shape
        zd = zd.reshape(-1, settings[key].nNodes, settings[key].deltas // 2 - 1)
        MIN, MAX = np.min(zd, axis = 1), np.max(zd, axis = 1)
        zd = np.array([(i - i.min()) / (i.max() - i.min()) for i in zd])
        zd[zd <= np.finfo(zd.dtype).eps] = 0
        zd[np.isfinite(zd) == False] = 0
        zd = zd.reshape(s)

    zds[key] = zd
    deltas = settings[key].deltas // 2 - 1

    LIMIT = deltas
    LIMIT = np.inf
    with mp.Pool(processes) as p:
        aucs_raw = np.array(\
                    p.map(\
                          worker, tqdm(zd.reshape(-1, settings[key].nNodes, deltas))\
                          )\
                     )

    rshape  = tuple(i for i in zd.shape if i != settings[key].deltas // 2 - 1)
    rshape += (2,) # add error
    tmp          = aucs_raw.reshape(rshape)
    aucs_raw     = tmp[..., 0]
    errors       = tmp[..., 1].mean(-2)

    s = aucs_raw.shape

    auc = aucs_raw.copy()
    rows, columns = s[1], s[0] - 1
    fig, ax  = plt.subplots(rows, columns, sharex = 'all')  # cleaned data
    sfig, sax = plt.subplots(rows, columns, sharex = 'all') #  outlier detection
    colors = plt.cm.tab20(range(settings[key].nNodes))
    conditions = np.fromiter(data[key].keys(), dtype = float)
    for row in range(rows): # nudge conditions
        for column in range(columns): # temperatures
            for n in range(s[-1]): # nodes
                tax = ax[row, column] if rows * columns > 1 else ax
                stax = sax[row, column] if rows * columns > 1  else sax
                tmpraw = aucs_raw[[0, column + 1], row, :, n]
                tmp = tmpraw.copy()
                try:
                    inliers, outliers = RejectOutliers(tmp.T)
                    inliermean = tmp[:, inliers].mean(-1)
                    for outlier in outliers:
                        tmp[:, outlier] = inliermean
                    tax.scatter(*tmp, color = colors[n])
    
                    if column == 0: # add label per row
                        rowLabel = f'M={conditions[row]}'
                        d = dict(fontsize = 15)
                        tax.text(.2, .8, rowLabel,\
                                 transform = tax.transAxes,\
                                 horizontalalignment = 'right',
                                 fontdict = d)
                        stax.text(.2, .8, rowLabel,\
                                  transform = stax.transAxes,\
                                  horizontalalignment = 'right',\
                                  fontdict = d)
                except:
                    continue

                stax.scatter(*tmpraw[:, inliers], color = colors[n])
                stax.scatter(*tmpraw[:, outliers], color = colors[n], marker = 's')

                auc[[0, column + 1], row, :, n] = tmp # update correction

    # make general legend
    elements = [plt.Line2D([0], [0], marker = 'o', linestyle = 'none', color = colors[i], label = j) \
                for i, j in settings[key].rmapping.items()]

    # add global axes for alignment
    mainax = fig.add_subplot(111, frameon = False, xticks = [], yticks = [])
    mainax.set_xlabel(information_impact, labelpad = 50)
    mainax.set_ylabel(causal_impact, labelpad = 50)
    mainax.legend(handles = elements, loc = 'upper left', \
                  bbox_to_anchor = (1,1), borderaxespad = 0, frameon = False,\
                  handletextpad = 0.01,\
                  borderpad = 0.01)
    fig.subplots_adjust(hspace = 0)

    # support figure
    mainax = sfig.add_subplot(111, frameon = False, xticks = [], yticks = [])
    mainax.set_xlabel(information_impact, labelpad = 50)
    mainax.set_ylabel(causal_impact, labelpad = 50)
    mainax.legend(handles = elements, loc = 'upper left', \
                  bbox_to_anchor = (1,1), borderaxespad = 0, frameon = False,\
                  handletextpad = 0.01,\
                  borderpad = 0.01)
    sfig.subplots_adjust(hspace = 0)
    mainax.set_title('Outlier rejection')

    # save figures
#l    fig.savefig(os.path.join(figDir, f'{key}_causalimpact_information_impact.eps'))
#    sfig.savefig(os.path.join(figDir, f'{key}_raw_causalimpact_information_impact.eps'))

    aucs[key] = auc # update data
# %%


for k, v in loadedData.items():
#    v = v.squeeze()
#    v = (v - v.min(0)) / (v.max(0) - v.min(0))
    
    colors = plt.cm.tab20(np.arange(settings[k].nNodes))
    fig, ax = plt.subplots()
    graph = nx.node_link_graph(settings[k].graph )
    tmp = sorted(nx.connected_component_subgraphs(graph), \
                 key = lambda x: len(x))
    
    inax = ax.inset_axes((0.3, 0.3, 1, 1))
    nx.draw(tmp[-1], ax = inax, pos = nx.circular_layout(tmp[-1]),  with_labels =1)
    s = v.shape
    
    tmp =  v.reshape(s[0], -1, s[-2], s[-1]).mean(1)
    [ax.plot(i, color = colors[idx]) for idx, i in enumerate(tmp[1])]    
    [ax.plot(i, linestyle ='--', color = colors[idx]) for idx, i in enumerate(tmp[0])]
#    ax.scatter(*tmp[[0, 1   ], ..., -1])
    colors = plt.cm.tab20(np.arange(settings[k].nNodes))
    elements = [plt.Line2D([0], [0], label = j, color = colors[idx]) for j, idx in \
                settings[k].mapping.items()]
    ax.legend(handles = elements)
#    ax.set_xlim(0, 5)
    ax.set_title(k)

# %%
    
        
    
plt.show()
# centralities = {
#                     'degree' : partial(nx.degree, weight = 'weight'), \
#                     'betweenness': partial(nx.betweenness_centrality, weight = 'weight'),\
#                     'current flow'  : partial(nx.information_centrality, weight = 'weight'),\
#                     'eigenvector'  : partial(nx.eigenvector_centrality, weight = 'weight'),\
#             }
#
#
# # produce ranking for mu with causal influence metrics
# ranking    = np.argsort(auc, axis = -1)
# prediction = (ranking[0] == ranking[[1, 2]])
# structural = np.zeros((len(centralities), setting.nNodes))
#
#
# for idx, (k, f) in enumerate(centralities.items()):
#     structural[idx] = np.argsort(np.fromiter(dict(f(setting.graph)).values(), dtype = float))
#
# pred_struct = np.array([i  == ranking[[1,2]].reshape(-1, setting.nNodes) for i in structural])
# pred_struct = pred_struct.reshape((len(centralities), *ranking[[1,2]].shape))
#
# pred = np.vstack((prediction[None, ...], pred_struct))
# tmp = pred[..., -1] # driver-estimates
#
# tmp = tmp.reshape((tmp.shape[0], 2, -1))




labels = ['informational\nimpact', *centralities.keys()]
conditions = 'Underwhelming Overwhelming'.split()
width = .5
x = np.arange(tmp.shape[0]) - width
fig, ax = plt.subplots(1)

elements = []
for idx in range(tmp.shape[1]):
    for jdx, xi in enumerate(x):
        print(tmp[jdx, idx].std())
        ax.bar(xi + idx * width, tmp[jdx, idx].mean(), color = colors[idx], width = width)
        ax.errorbar(xi + idx * width, tmp[jdx, idx].mean(), tmp[jdx, idx].std(), \
                    color = 'black', capsize = 10, capthick = 2)
    ax.set_xticks(x + width)
    ax.set_xticklabels(labels)

    element = plt.Line2D([0], [0], linestyle = 'none', marker = 's', label = conditions[idx], color = colors[idx])
    elements.append(element)

ax.legend(handles = elements, loc = 'upper right')
ax.set_ylabel('Prediction accuracy (ratio)')
