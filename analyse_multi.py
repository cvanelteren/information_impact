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
root = '/home/casper/projects/information_impact/Data/1548025318.5751357'

data     = IO.DataLoader(root) # extracts data folders
settings = {key : IO.Settings(root) for key in data} # load corresponding settings

centralities = {
                    r'$c_i^{deg}$' : partial(nx.degree, weight = 'weight'), \
                    r'$c_i^{betw}$': partial(nx.betweenness_centrality, weight = 'weight'),\
                    r'$c_i^{ic}$'  : partial(nx.information_centrality, weight = 'weight'),\
                    r'$c_i^{ev}$'  : partial(nx.eigenvector_centrality, weight = 'weight'),\
            }

figDir = '../thesis/figures/'
information_impact = '$\mu_i$'
causal_impact      = '$\gamma_i$'
# %%
# begin the uggliness
class Worker:
    def __init__(self , datadict, setting):
        self.setting = setting
        self.filenames = sorted(\
                           misc.flattenDict(datadict), \
                           key = lambda x: os.path.getctime(x), \
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

        fileName  = self.filenames[fidx]
        fileNames = self.filenames
        setting = self.setting
        # do control
        data = np.frombuffer(self.buff).reshape(self.buffshape)
        node, temp, trial = np.unravel_index(fidx, self.expectedShape, order = 'F')
        # load settings
        t     = fileName.split('/')[-4]
        setting = settings[t] # load from global ; bad


#        DELTAS_, EXTRA = divmod(setting.deltas, 2) # extract relevant time data
#        DELTAS  = DELTAS_ - 1
        # TODO: check  this
        # only use half of the data

        # control data
        if '{}' in fileName:
            # load control; bias correct mi
            control = IO.loadData(fileName)
            # panzeri-treves correction
            mi   = control.mi
            bias = stats.panzeriTrevesCorrection(control.px,\
                                                 control.conditional, \
                                                 setting.repeat)
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
            control = IO.loadData(fileNames[controlidx])
             # load nudge
            sample  = IO.loadData(fileName)
            # impact  = stats.KL(control.px, sample.px)
            impact = stats.KL(sample.px, control.px)
            # don't use +1 as the nudge has no effect at zero
            redIm = np.nansum(impact[-self.deltas:], axis = -1).T
            # TODO: check if this works with tuples (not sure)
            for name in nodeNames:
                data[(node - 1) // setting.nNodes + 1, trial, temp, name,  :] = redIm.squeeze().T
        self.pbar.update(1)
# %% load data
# functionize this; part of IO?
loadedData = {}
for key in data:
    setting = settings[key]
    processes = mp.cpu_count()
    worker = Worker(data[key], setting)
    from multiprocessing.pool import ThreadPool
    
    with ThreadPool(processes = processes) as p:
        p.map(worker, worker.idx)
    loadedData[key] = np.frombuffer(worker.buff, dtype = np.float64).reshape(*worker.buffshape)
del worker
# %% plot graphs
colors = plt.cm.tab20(np.arange(12))
plt.rcParams['axes.prop_cycle'] = plt.cycler('color', colors)
plt.style.use('seaborn-poster')

from matplotlib.patches import Circle

# plot pretty graphs with centralities
for k, v in loadedData.items():
    fig, ax = plt.subplots()
    ax.plot(v[1, 0].mean(0).T)

    fig, ax  = plt.subplots(len(centralities) // 2, 2)
    graph    = nx.readwrite.json_graph.node_link_graph(settings[k].graph)
    pos      = nx.nx_agraph.graphviz_layout(graph, prog = 'neato', \
                                        )
    for idx, (cent, cf) in enumerate(centralities.items()):
        c = dict(cf(graph))
        s = np.fromiter(c.values(), dtype = float)
        s = (s - s.min()) /(s.max() - s.min())
        tax = ax[:, :].ravel()[idx]
        tax.axis('off')
        tax.set_aspect('equal','box')

        centralityLabel = \
        cf.func.__name__.replace('_', ' ').replace('centrality','')

        tax.set_title(centralityLabel)

        plotz.addGraphPretty(graph, tax, pos, \
                         mapping = settings[k].mapping,\
                         )
        for pidx, pp in enumerate(tax.get_children()):
            if isinstance(pp, Circle):
                pp.set(radius = s[pidx] * pp.radius * 2.4 )
    fig.subplots_adjust(hspace = .05, wspace = 0)
    fig.savefig(os.path.join(figDir, f'{k}_centrality.eps'))
    


# %%
# fit functions
double = lambda x, a, b, c, d, e, f: a + b * np.exp(-c*(x)) + d * np.exp(- e * (x-f))
double_= lambda x, b, c, d, e, f: b * np.exp(-c*(x)) + d * np.exp(- e * (x-f))
single = lambda x, a, b, c : a + b * np.exp(-c * x)
single_= lambda x, b, c : b * np.exp(-c * x)
special= lambda x, a, b, c, d: a  + b * np.exp(- (x)**c - d)

func        = double
p0          = np.ones((func.__code__.co_argcount - 1)); # p0[0] = 0
fitParam    = dict(maxfev = int(1e6), \
                   bounds = (0, np.inf), p0 = p0,\
                   jac = 'cs')

# %% normalize data

# uggly mp case
# note use of globals here!
def worker(sample):
    auc = np.zeros((len(sample), 2))
    coeffs, errors = plotz.fit(sample, func, params = fitParam)
    for nodei, c in enumerate(coeffs):
        tmp = 0
        F   = lambda x: func(x, *c) - c[0]
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
    
    conditions = np.fromiter(data[key].keys(), dtype = float)
    for row in range(rows): # nudge conditions
        for column in range(columns): # temperatures
            for n in range(s[-1]): # nodes
                tax = ax[row, column]
                stax = sax[row, column]
                tmpraw = aucs_raw[[0, column + 1], row, :, n]
                tmp = tmpraw.copy()
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
    fig.savefig(os.path.join(figDir, f'{key}_causalimpact_information_impact.eps'))
    sfig.savefig(os.path.join(figDir, f'{key}_raw_causalimpact_information_impact.eps'))
    
    aucs[key] = auc # update data
# %%

for i in zd:
    for j in i:
        fig, ax = plt.subplots()
        for c, k in zip(colors, j.mean(0)):
            ax.plot(k, color = c)
# %%

for l in nx.generate_multiline_adjlist(graph):
    tmp = []
    w = l.split()
    try:
        for i in w:
            print(i)
            tmp.append(literal_eval(i))
    except:
        tmp.append(w)
    print(tmp)
