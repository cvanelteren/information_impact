#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 18 12:09:36 2018

@author: casper
"""
from numpy import *
from functools import partial

from scipy import optimize, integrate
from matplotlib.pyplot import *
from time import sleep
from Utils import plotting as plotz, stats, IO
import os, re, networkx as nx, scipy, multiprocessing as mp
warnings.filterwarnings('ignore')
close('all')
style.use('seaborn-poster')
dataPath = f"{os.getcwd()}/Data/"

# dataPath = '/run/media/casper/test/'
#dataPath = '/mnt/'
# convenience
kite     = '1548347769.6300871'
psycho   = '1548025318.5751357'
# multiple = '1550482875.0001953'

extractThis      = IO.newest(dataPath)[-1]
extractThis      = psycho
extractThis      = kite
#extractThis      = '1547303564.8185222'
#extractThis  = '1548338989.260526'
extractThis = extractThis.split('/')[-1] if extractThis.startswith('/') else extractThis
loadThis    = extractThis if extractThis.startswith('/') else f"{dataPath}{extractThis}"
data        = IO.DataLoader(loadThis)[extractThis]

settings = IO.Settings(loadThis)
deltas   = settings.deltas
repeats  = settings.repeat


temps    = [float(i.split('=')[-1]) for i in data.keys()]
nudges   = list(data[next(iter(data))].keys())
temp     = temps[0]

# Draw graph ; assumes the simulation is over 1 type of graph


graph     = settings.graph
model     = settings.loadModel() # TODO: replace this with mappings
# control  = IO.loadData(data[f't={temp}']['control'][0]) # TEMP WORKAROUND
# model    = Ising(control.graph)
NTRIALS  = settings.nTrials
NTEMPS   = len(data)
NNUDGE   = len(settings.pulseSizes) + 1
NODES    = settings.nNodes

DELTAS_, EXTRA = divmod(deltas, 2) #use half, also correct for border artefact
COND     = NNUDGE - 1
DELTAS   = DELTAS_ - 1

pulseSizes = settings.pulseSizes

print(f'Listing temps: {temps}')
print(f'Listing nudges: {pulseSizes}')

figDir = f'../thesis/presentation/figures/{extractThis}'
# %% # show mag vs temperature
func = lambda x, a, b, c, d :  a / (1 + exp(b * (x - c))) + d # tanh(-a * x)* b + c
for root, subdirs, filenames in os.walk(loadThis):
    msettings = {}

    if any(['mags' in i for i in filenames]):
        msettings = IO.loadPickle(os.path.join(root, 'mags.pickle'))

    if msettings:
        # fitted
        fx = msettings.get('fitTemps', msettings.get('temps')) # legacy
        fy = msettings.get('fmag', msettings['mag'])

        # matched
        mx = msettings.get('matchedTemps', msettings.get('temperatures')) # legacy
        my = msettings.get('magRange')

        fig, ax  = subplots(figsize = (10, 12))
        ax.scatter(fx, fy, alpha = .2)
        ax.scatter(mx, my * fy.max(), color = 'red', zorder = 2)
        a, b = scipy.optimize.curve_fit(func, fx, fy, maxfev = 10000)
        x = linspace(fx.min(), fx.max(), 1000)
        ax.plot(x, func(x, *a), '--k')
        ax.set_xlabel('Temperature (T)', fontsize =30)
        ax.set_ylabel('|<M>|',  fontsize = 30)
        rcParams['axes.labelpad'] = 10
        fig.savefig(figDir + f"temp_mag.png", bbox_inches = 'tight', pad_inches = 0)
#assert 0
# %% plot graph

centralities = {
                    r'$c_i^{deg}$' : partial(nx.degree, weight = 'weight'), \
                    r'$c_i^{betw}$': partial(nx.betweenness_centrality, weight = 'weight'),\
                    r'$c_i^{ic}$'  : partial(nx.information_centrality, weight = 'weight'),\
                    r'$c_i^{ev}$'  : partial(nx.eigenvector_centrality, weight = 'weight'),\
             }

# %%
colors = cm.tab20(arange(NODES))
fig, ax  = subplots(figsize = (10, 10), frameon = False)
ax.set(xticks = [], yticks = [])
positions = nx.nx_agraph.graphviz_layout(model.graph, prog = 'neato', \
                                         root = 'happy')
#positions = nx.shell_layout(graph)
#positions = nx.nx_pydot.pydot_layout(graph, prog = '')
#                                         root = sorted(dict(model.graph.degree()).items())[0][0])

props = dict(
         annotate = dict(fontsize  = 1.3, annotate = True),\
         )
#p = {}
plotz.addGraphPretty(model.graph, ax = ax, positions = positions, \
                     mapping = model.mapping,
                     **props,\
                     )


#nx.draw(model.graph, positions, ax = ax)
ax.set_aspect('equal', 'box')
ax.set(xticks = [], yticks = [])
ax.axis('off')
fig.show()
savefig(figDir + 'network.png',\
        bbox_inches = 'tight', pad_inches = 0, transparent = True)



#%%
centLabels = 'Degree Betweenness Information Eigenvector'.split()
idx = 22

props['annotate']['fontsize'] = 1.9
fig, ax = subplots(2, 2, figsize = (idx, idx))
from matplotlib.patches import Circle
#fig.subplots_adjust(hspace = 0.07, wspace = 0, left = 0, right = 1)
for idx, (cent, cf) in enumerate(centralities.items()):
    c = dict(cf(model.graph))
    s = array(list(c.values()))
    s = (s - s.min()) /(s.max() - s.min()) 
    tax = ax[:, :].ravel()[idx]
    tax.axis('off')
#    tax.set_aspect('equal','box')
    tax.set_title(centLabels[idx], fontsize = 40, color = 'white')
    plotz.addGraphPretty(model.graph, tax, positions, \
                     mapping = model.mapping,\
                     **props,\
                     )
    for artist in tax.get_children():
        if isinstance(artist, Circle):
            lab = artist.get_label()
            lab = int(lab) if lab.isdigit() else lab
            pidx = model.mapping[lab]
            tmp  = (s[pidx]) * artist.radius 
            tax.add_artist(Circle(artist.center, facecolor = artist.get_facecolor(), radius = tmp))
            artist.set(facecolor = 'none')
    
#            artist.set(alpha = s[pidx])
#mainax = fig.add_subplot(111, xticks = [], yticks = [], frameon = False)
#mainax.legend(handles = [Line2D([0],[0], color = colors[idx], marker = 'o', linestyle = 'none',\
#                                label = node) for idx, node in enumerate(graph)], \
#             bbox_to_anchor = (1, 1), loc = 'upper left', borderaxespad = 0)
#for item in [fig, ax]:
#    item.set_visisble(False)
fig.subplots_adjust(hspace = 0, wspace = 0)
fig.savefig(figDir +  'graph_and_cent.png', \
            bbox_inches = 'tight', pad_inches = 0, transparent = True)
fig. show()
#assert 0
# %%
fig, ax = subplots(1, 2)
props['annotate']['annotate'] = False
for idx, tax in enumerate(ax):
    cf = centralities.values()[idx]
    c = dict(cf(model.graph))
    s = array(list(c.values()))
    s = (s - s.min()) /(s.max() - s.min()) 
    tax = ax[:, :].ravel()[idx]
    plotz.addGraphPretty(graph, ax = tax, positions = positions, \
                         mapping = model.mapping,\
                         **props)
    for artist in tax.get_children():
        if isinstance(artist, Circle):
            lab = artist.get_label()
            lab = int(lab) if lab.isdigit() else lab
            pidx = model.mapping[lab]
            tmp  = (s[pidx]) * artist.radius 
            tax.add_artist(Circle(artist.center, facecolor = artist.get_facecolor(), radius = tmp))
            artist.set(facecolor = 'none')
    tax.axis('off')
fig.subplots_adjust(wspace = 0, hspace = 0)
fig.savefig(figDir + 'cent_simple.png',pad_inches = 0,\
        bbox_inches = 'tight')
assert 0
# %%
# extract data for all nodes
information_impact = '$\mu_i$'
causal_impact      = '$\gamma_i$'
from tqdm import tqdm


# define color swaps

# swap default colors to one with more range
rcParams['axes.prop_cycle'] = cycler('color', colors)

# %% Need an alternative for this badly...
def flattenDict(d):
    out = []
    for k, v in d.items():
        if isinstance(v, dict):
            out += flattenDict(v)
        else:
            out.append(v)
    return out

def worker(fidx):
    """
    Temporary worker to load data files
    This function does the actual processing of the correct target values
    used in the analysis below
    """
    fileName = fileNames[fidx]
    # do control
    data = frombuffer(var_dict.get('X')).reshape(var_dict['xShape'])
    node, temp, trial = unravel_index(fidx, var_dict.get('start'), order = 'F')

    # control data
    if '{}' in fileName:
        # load control; bias correct mi
        control = IO.loadData(fileName)
        # panzeri-treves correction
        mi   = control.mi
        bias = stats.panzeriTrevesCorrection(control.px,\
                                             control.conditional, \
                                             repeats)
        # mi -= bias

        data[0, trial, temp]  = mi[:DELTAS, :].T
    # nudged data
    else:
        targetName = fileName.split('=')[-1].strip('.pickle') # extract relevant part
        # extract pulse and only get left part
        targetName = re.sub('{|}', '', targetName).split(':')[0]
        # get the idx of the node
        
        nodeNames = []
        for name, idx in model.mapping.items():
            if str(name) in targetName:
                nodeNames.append(idx)
                
        # assert model.rmapping[nodeNames[0]] in targetName, f'node not matched {targetName} model.rmapping[nodeNames[0]]'
        # print(targetName, model.mapping[nodeNames[0]], mp.current_process())
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
        redIm = nansum(impact[-DELTAS:], axis = -1).T
        # TODO: check if this works with tuples (not sure)
        for name in nodeNames:
            data[(node - 1) // model.nNodes + 1, trial, temp, name,  :] = redIm.squeeze().T

# look for stored data [faster]
fasterData = f'results.pickle'
try:
    for k, v in IO.loadPickle(fasterData):
        globals()[k] = v
    NNUDGE, NTEMPS, NTRIALS, NODES, DELTAS = loadedData.shape

# start loading individual pickles
except:
    fileNames = sorted(\
                       [j for i in flattenDict(data) for j in i],\
                       key = lambda x: os.path.getmtime(x),\
                       )
#    fileNames = [j for i in flattenDict(data) for j in i]
    var_dict = {}
    def initWorker(X, xShape, start):
        var_dict['X']      = X
        var_dict['xShape'] = xShape
        var_dict['start']  = start
    expStruct = (NODES * COND  + 1, NTRIALS, NNUDGE)
    buffShape = (NNUDGE, NTEMPS, NTRIALS, NODES, DELTAS)
    buff = mp.RawArray('d', int(prod(buffShape)))

    # for intermitten dataviewing only load complete sets otherwise nans
    a, b = divmod(len(fileNames), COND * NODES + 1 ) # extract full divisors
    c = 1 if b else 0 # subtract 1 if not full set
    tidx = (COND * NODES + 1) * (a - c)
#    print(a, b, c, COND * NODES + 1 )
    print(f'Loading {tidx}')
    tmp = range(tidx)
    processes = mp.cpu_count()
    with mp.Pool(processes = processes, initializer = initWorker,\
              initargs = (buff, buffShape, expStruct)) as p:
        p.map(worker, tqdm(tmp))
    loadedData = frombuffer(buff, dtype = float64).reshape(*buffShape)
    # store for later
#    IO.savePickle(fasterData, dict(\
#                  loadedData = loadedData, data = data))
    del buff
# %% extract root from samples

# fit functions
double = lambda x, a, b, c, d, e, f: a + b * exp(-c*(x)) + d * exp(- e * (x-f))
double_= lambda x, b, c, d, e, f: b * exp(-c*(x)) + d * exp(- e * (x-f))
single = lambda x, a, b, c : a + b * exp(-c * x)
single_= lambda x, b, c : b * exp(-c * x)
special= lambda x, a, b, c, d: a  + b * exp(- (x)**c - d)

import sympy as sy
from sympy import abc
from sympy.utilities.lambdify import lambdify
symbols     = (abc.x, abc.a, \
               abc.b, abc.c, \
               abc.d, abc.e,\
               abc.f)\

syF         = symbols[1] * sy.exp(- symbols[2] * \
                     symbols[0]) +\
                     symbols[3] * sy.exp(- symbols[4] * \
                            (symbols[0] - symbols[5]))
#print(syF)
##syF         = symbols[1] * sy.exp(- symbols[2] * symbols[0])
#func        = lambdify(symbols, syF, 'numpy')
func        = double
p0          = ones((func.__code__.co_argcount - 1)); # p0[0] = 0
fitParam    = dict(maxfev = int(1e6), \
                   bounds = (0, inf), p0 = p0,\
                   jac = 'cs')

repeats  = settings.repeat

# %% normalize data

from scipy import ndimage
zd = zeros(loadedData.shape)
for temp in range(NTEMPS):
    for nudge in range(NNUDGE):
        zdi = loadedData[nudge, temp]
        # scale data 0-1 along each sample (nodes x delta)
        rescale = True
#        rescale = False

#        rescale for each trial over min / max
#        zdi = ndimage.filters.gaussian_filter1d(zdi, 8, axis = -1)
#        zdi = ndimage.filters.gaussian_filter1d(zdi, 3, axis = 0)
        if rescale:
            zdi = zdi.reshape(NTRIALS, -1) # flatten over trials
            MIN, MAX = nanmin(zdi, axis = 1), nanmax(zdi, 1)
            MIN = MIN[:, newaxis]
            MAX = MAX[:, newaxis]
            zdi = (zdi - MIN) / (MAX - MIN)
            zdi = zdi.reshape((NTRIALS, NODES, DELTAS))

        thresh = 1e-4
        #zd[zd <= thresh] = 0
        zdi[zdi <= finfo(zdi.dtype).eps] = 0 # remove everything below machine error
#        zdi[zdi < 0]  = 0
        # show means with spread
        # remove the negative small numbers, e.g. -1e-5
        zdi[isfinite(zdi) == False] = 0 # check this
        zd[nudge, temp] = zdi

# %% time plots

# insert dummy axis to offset subplots
gs = dict(\
          height_ratios = [1, 1, 1], width_ratios = [1, .15, 1, 1],\
          )
fig, ax = subplots(3, 4, sharey = 'all', sharex = 'all',  gridspec_kw = gs)
mainax  = fig.add_subplot(111, frameon = 0)
mainax.set(xticks = [], yticks = [])

#mainax.set_title(f'\n\n').
mainax.set_xlabel('Time (t)', fontsize = 22, labelpad = 40)

x  = arange(DELTAS)
xx = linspace(0, 1 * DELTAS, 1000)
sidx = 2 # 1.96
labels = 'Unperturbed\tUnderwhelming\tOverwhelming'.split('\t')
_ii    = '$I(s_i^{t_0 + t} : S^{t_0})$'

[i.axis('off') for i in ax[:, 1]]

rcParams['axes.labelpad'] = 80
ax[1, 2].set_ylabel("$D_{KL}(P' || P)$", fontsize = 22, labelpad = 5)
ax[1, 0].set_ylabel(_ii, fontsize = 22, labelpad = 5)
from matplotlib.ticker import FormatStrFormatter
means, stds  = nanmean(zd, -3), nanstd(zd, -3)

for temp in range(NTEMPS):
    for nudge in range(NNUDGE):
        idx = nudge if nudge == 0 else nudge + 1
        tax = ax[temp, idx]
        if temp == 0:
            tax.set_title(labels[nudge] + '\n')
        # compute mean fit
        mus    = means[nudge, temp]
        sigmas = stds[nudge, temp]
        
        
        meanCoeffs, meanErrors = plotz.fit(mus, func, params = fitParam)
        
        tmpauc = [scipy.integrate.quad(lambda x: func(x, *c), 0, np.inf)[0] for c in meanCoeffs]
        leader = np.argmax(tmpauc)
#        print(np.argsort(tmpauc))
        for node, idx in sorted(model.mapping.items(), key = lambda x : x[1]):
            # plot mean fit
            # tax.plot(xx, func(xx, *meanCoeffs[idx]),\
            #          color = colors[idx],\
            #          alpha = 1, \
            #          markeredgecolor = 'black',\
            #          label =  node)
            # plot the raw data
            zorder = 1 if idx != leader else 5
            # alpha  = .1 if zorder != 5 else 0
            # tax.fill_between(x, mus[idx] - sidx * sigmas[idx], mus[idx] + sidx * sigmas[idx],\
                              # color = (1 - alpha) * colors[idx],zorder = zorder)
            tax.errorbar(x, mus[idx],\
                              fmt = '-',\
                              yerr = sidx * sigmas[idx],\
                              capthick = 500,\
                              markersize = 15, \
                              color = colors[idx],\
                              label = node, \
                              alpha = 1 if zorder == 1 else 1) # mpl3 broke legends?
            if idx == leader:
                xx, yy = (deltas // 2  * .90, .8)
                arti = Line2D([xx], [yy], marker = 'o', color = colors[idx],\
                       )
                tax.add_artist(arti)
#                tax.annotate('Driver-node', (*x,*y), horizontalalignment = 'right')

           


        tax.ticklabel_format(axis = 'y', style = 'sci',\
                         scilimits = (0, 4))

    tax.text(.95, .9, \
             f'$M_r$ = {round(temps[temp], 2)}', \
             fontdict = dict(fontsize = 15),\
             transform = tax.transAxes,\
             horizontalalignment = 'right')
#    tax.set(xlim = (-1.5, 30))
    # if temp  == 0 and nudge == 0 :
        # print('HERHERHERH')

# format plot
h = [Line2D([0], [0], marker = 'o', linestyle = 'none',\
            color = colors[idx], label = node) for idx, node in model.rmapping.items()]
mainax.legend(\
              handles        = h,\
              title          = 'Node', \
              title_fontsize = 15, \
              loc            = 'upper left', \
              bbox_to_anchor = (1, 1),\
              borderaxespad  = -.1,\
              handletextpad = -.01,\
              frameon        = False,\
              )
subplots_adjust(hspace = 0, wspace = 0)

fig.show()
fig.savefig(figDir + 'mi_time.eps', pad_inches = 0,\
        bbox_inches = 'tight')

#assert 0 
# %% presentation plot
rcParams['axes.labelpad'] = 0
fig, ax = subplots(figsize = (15, 10))
x  = arange(DELTAS)
xx = linspace(0, 1 * DELTAS, 1000)
y  = means[0,0]
coeffs, errors = plotz.fit(y, func, params = fitParam)
for node in range(NODES):
    ax.scatter(x, y[node], label = model.rmapping[node], \
               color = colors[node])
    ax.plot(xx, func(xx, *coeffs[node]), color = colors[node])
    if model.rmapping[node] == 8 or model.rmapping[node] == 5:
        tmp = lambda x : func(x, *coeffs[node]) - .5 * y[node][0]
        yy  =optimize.fsolve(tmp, 0)
        idx = argmin(abs(x - yy))
        print(y[node][idx])
        ax.axvline(yy, 0, y[node][idx], color = colors[node],\
                   linestyle = '-', zorder = 5)
ax.axhline(.05, color = 'k', linestyle = 'dashed')
ax.set_ylabel('$I(s_i^{t_0 + t} ; S^t_0)$', fontsize = 50)
ax.set_xlabel('$time [t]$', fontsize = 50)
ax.tick_params(labelsize = 25)
ax.legend(title = 'Node', title_fontsize = 40, fontsize = 25)
fig.savefig('/home/casper/projects/thesis/presentation/figures/mi_example.eps')
# %% estimate impact
#aucs       = zeros((NTRIALS, COND, NODES))

from scipy import interpolate
from time import sleep

COEFFS = zeros((COND, NTRIALS, NODES, p0.size))
x = arange(DELTAS)

# uggly mp case
LIMIT = DELTAS
LIMIT = inf
from sklearn import metrics
def worker(sample):
    auc = zeros((len(sample), 2))
    coeffs, errors = plotz.fit(sample, func, params = fitParam)
    for nodei, c in enumerate(coeffs):
        tmp = 0
#        if c[0] < 1e-4:
#            tmp = syF.subs([(s, v) for s,v in zip(symbols[1:], c)])
#            tmp = sy.integrals.integrate(tmp, (abc.x, 0, sy.oo),\
#                                     meijerg = False).evalf()
        F   = lambda x: func(x, *c) - c[0]
#        tmp = metrics.auc(F)
        tmp, _ = scipy.integrate.quad(F, 0, LIMIT)
        auc[nodei, 0] = tmp
        auc[nodei, 1] = errors[nodei]
    return auc

import multiprocessing as mp
with mp.Pool(processes) as p:
    aucs_raw = array(\
                p.map(\
                      worker, tqdm(zd.reshape(-1, NODES, DELTAS))\
                      )\
                 )

rshape  = tuple(i for i in loadedData.shape if i != DELTAS)
rshape += (2,) # add error
tmp          = aucs_raw.reshape(rshape)
aucs_raw     = tmp[..., 0]
errors       = tmp[..., 1].mean(-2)

# remove machine error estimates
aucs_raw[aucs_raw <= finfo(aucs_raw.dtype).eps / 2] = 0

# %% show idt auc vs impact auc and apply correction

from sklearn.covariance import EllipticEnvelope
from sklearn.neighbors import LocalOutlierFactor as lof
from sklearn.covariance import MinCovDet
rcParams['axes.labelpad'] = 40
fig, ax = subplots(3, 2, sharex = 'col')
subplots_adjust(hspace = 0, wspace = .2)
mainax = fig.add_subplot(111, frameon = False, \
                         xticks = [], yticks = [],\
                         xlabel = f'Information impact ({information_impact})',\
                         ylabel = f'Causal impact ({causal_impact})'\
                         )
out = lof(n_neighbors = NTRIALS)

aucs   = aucs_raw.copy()
thresh = 2.5
clf    = MinCovDet()

labels       = 'Underwhelming\tOverwhelming'.split('\t')
rejections   = zeros((COND, NTEMPS, NODES))
showOutliers = True
#showOutliers = False
pval         = .01

pcorr = (NNUDGE - 1) * NTEMPS
from sklearn.linear_model import Ridge
from sklearn.feature_selection import f_regression
ridge = Ridge()

for temp in range(NTEMPS):
    for nudge in range(1, NNUDGE):
        tax = ax[temp, nudge - 1]
#        tax.set_aspect('equal', 'box')
        smax = aucs[[0, nudge], temp].ravel().max(0)
        smin = aucs[[0, nudge], temp].ravel().min(0)
        ranges = linspace(-1, smax, 100)
#
        xx, yy = meshgrid(ranges, ranges)

        for node, idx in model.mapping.items():

            # value error for zero variance
#            try:
            tmp = aucs_raw[[0, nudge], temp, :, idx]
            clf.fit(tmp.T)
            Z = clf.mahalanobis(np.c_[xx.ravel(), yy.ravel()])
            
            
            if showOutliers:
                print('plotting outline')
                tax.contour(xx, yy, sqrt(Z.reshape(xx.shape)), \
                        colors = colors[[idx]],\
                        levels = [thresh], linewidths = 2, alpha = 1, zorder = 5)

            # plot ci vs ii
            ident = sqrt(clf.mahalanobis(tmp.T))
#            print(model.rmapping[idx], ident)
            outliers = where(ident >= thresh)[0]
            ingroup  = where(ident < thresh)[0]
            rejections[nudge - 1, temp, idx] = len(outliers) / NTRIALS
            ingroupmean = tmp[:, ingroup].mean(1)
            for outlier in outliers:
                aucs[[0, nudge], temp, outlier, idx]= ingroupmean
                if showOutliers:
                    tax.scatter(*aucs_raw[[0, nudge], temp, outlier, idx], \
                            color = colors[idx], \
                            alpha = 1, marker = 's')
#            except:
#                pass.

#            tax.scatter(*tmp.mean(1), color = 'k', s = 50, zorder = 5)
            alpha = .1 if showOutliers else 1
            tax.scatter(*aucs[[0, nudge], temp, :, idx], \
                        color = colors[idx], \
                        label = node, alpha = alpha, \
                        linewidth = 1)
#            xx, yy = aucs[[0, nudge], temp].reshape(2, -1).min(1), aucs[[0, nudge], temp].reshape(2, -1).max(1)
            
            # tt = np.linspace(xx.min(), yy.min())
            # tax.plot(tt, tt, '--k')
#            tax.scatter(*tmp[:, ident < 0], s = 100, marker = 's')
#        slope, intercept, r, p, stder = scipy.stats.linregress(\
#                                       *aucs[[0, nudge], temp].reshape(COND, -1))

#        slope, intercept = scipy.stats.siegelslopes(\
#                    *aucs[[0, nudge], temp].reshape(COND, -1))

#        tx, ty = aucs[[0, nudge], temp].reshape(2, -1).copy()
#        ddof = tx.size - 2
#
#        ridge.fit(tx[:, None], ty)
#
#        p = 2 * ( 1 - scipy.stats.t.cdf(abs(ridge.coef_), ddof) )

        smax = aucs[[0, nudge], temp].ravel().max(0)
        smin = aucs[[0, nudge], temp].ravel().min(0)

#        r = ridge.score(tx[:, None], ty)

#        f, p = f_regression(tx[:, None], ty)
#        r, p  =scipy.stats.kendalltau(tx, ty)
#        rsq = r ** 2
#        rsq = r**2
        # plot regression
#        print(r, p )
#        if p < pval / pcorr:
##            print(p, )
#            tx = linspace(smin, smax)
##            ty = ridge.predict(tx[:, None]) * .8
#            ty = slope * tx + intercept
#            tax.plot(tx, ty, color = '#a2a2aa', linestyle = '--', alpha = 1)
#            tax.text(1, .1, \
#             f'$R^2$={rsq:.1e}\np={p:.1e}', \
#             fontdict  = dict(fontsize = 10),\
#             transform = tax.transAxes,\
#             horizontalalignment = 'right', \
#             verticalalignment   = 'bottom')
        if temp == 0:
            tax.set_title(labels[nudge-1])
            if nudge == NNUDGE - 1:
                leg = tax.legend(loc = 'upper left', bbox_to_anchor = (1, 1),\
                           frameon = False, title = 'Node',
                           borderaxespad = -.1,\
                           handletextpad = 0,\
                           title_fontsize = 15)
                # force alpha
                for l in leg.legendHandles:
                    l._facecolors[:, -1] = 1
        if nudge == 1:
             tax.text(.05, .9, \
             f'$M_r$ = {round(temps[temp], 2)}', \
             fontdict = dict(fontsize = 15),\
             transform = tax.transAxes,\
             horizontalalignment = 'left')
#        tax.set(yscale = 'log')


#        tax.set(yscale = 'log').

out = 'causal_ii_scatter_raw' if showOutliers else 'causal_ii_scatter_corrected'
out += '.png' if showOutliers else '.eps'
savefig(figDir + out, dpi = 1000)
# %% information impact vs centrality


centLabels = list(centralities.keys())
infImpact = aucs.mean(-2)

# get the test statistic
from mpl_toolkits.mplot3d import Axes3D
#ranking.sort()

conditionLabels = 'Underwhelming Overwhelming'.split()

#centralities['$c_i^{deg}$'] = nx.degree_centrality
centralities['$c_i^{deg}$'] = partial(nx.degree, weight = 'weight')
# plot info
rcParams['axes.labelpad'] = 20
for condition, condLabel in enumerate(conditionLabels):
    fig, ax = subplots(len(centralities), 3, sharex = 'all', sharey = 'row')
    mainax = fig.add_subplot(111, frameon = False, xticks = [], yticks = [])
    mainax.set_xlabel(f'Causal impact ({causal_impact})', labelpad = 50)
    for i, (cent, cent_func) in enumerate(centralities .items()):
        # compute cents
        centrality = dict(cent_func(model.graph))
#        rankCentrality[i] = argsort(centrality)[-1]
#        rankCentrality = zeros((len(centralities)))
        for temp in range(NTEMPS):
            tax = ax[i, temp]
#            tax.set(xscale = 'log')
            # scale node according to information impact
            tmp = infImpact[0, temp]
            tmp = (tmp - tmp.min()) / (tmp.max() - tmp.min())
            for node in range(NODES):
                nl = model.rmapping[node]
                tax.scatter(\
                            infImpact[condition + 1, temp, node], \
                            centrality[nl], \
                            color = colors[node], \
                            label = model.rmapping[node],\
                            s = max((600  * tmp[node], 10)), alpha = 1)
            # every column set title
            if temp == 0:
                tax.text(-.35, .5, cent, fontdict = dict(\
                                                  fontsize = 17),\
                     transform = tax.transAxes,\
                     horizontalalignment = 'center')
            # every row set label
            if i == 0:
                tax.set_title(f'$M_r$ = {round(temps[temp], 2)}')
                if temp == NTEMPS - 1:
                    leg = tax.legend(loc = 'upper left', \
                              bbox_to_anchor = (1.0, 1),\
                              borderaxespad = 0, frameon = False,\
                              title = 'Node',\
                              title_fontsize = 15, fontsize = 12)


    # hack for uniform labels
    for l in leg.legendHandles:
        l._sizes = [150]
    mainax.set_title(condLabel + '\n\n')
    #tax.legend(['Information impact', 'Causal impact'], loc = 'upper left', \
    #  bbox_to_anchor = (1.01, 1), borderaxespad = 0)
    subplots_adjust(wspace = .1, hspace = .1)
    #ax.set(xscale = 'log')
    savestr = figDir + f'causal_cent_ii_{conditionLabels[condition]}.eps'
    print(f'saving {savestr}')
    fig.savefig(figDir + f'causal_cent_ii_{conditionLabels[condition]}.eps')


# assert False
# %% driver node estimates

drivers = aucs.argmax(-1)
all_drivers = drivers.reshape(-1, NTRIALS)
N = len(centralities)
centApprox = zeros((NTEMPS, N, COND))


percentages = zeros((NTEMPS, N + 1, COND))

estimates = zeros((NTEMPS, NTRIALS,  N + 1, NODES))
maxestimates = zeros((NTEMPS, NTRIALS, N + 1), dtype = int)
targets   = zeros((NTEMPS, NTRIALS, COND, NODES))
for temp in range(NTEMPS):
    for cond in range(COND):
        percentages[temp, 0, cond] = equal(drivers[0, temp], \
                                   drivers[cond + 1, temp, ]).mean()

        estimates[temp, :, 0, :]   = aucs[0, temp]
        targets[temp, :, cond]     = aucs[cond + 1, temp]
        maxestimates[temp, :, 0]   = aucs[0, temp].argmax(-1)
        for ni, (cent, centF) in enumerate(centralities.items()):
            tmp = dict(centF(model.graph))
            centLabel, centVal= sorted(tmp.items(), \
                                       key = lambda x: abs(x[1]))[-1]


            centEstimate = model.mapping[centLabel]
#            print(ni + 1, centLabel, centEstimate)
            percentages[temp, ni + 1, cond] =  equal(\
                       centEstimate * ones(NTRIALS), \
                       drivers[cond + 1, temp]).mean()
#            print(cent, centVal)
            maxestimates[temp, :, ni + 1] = model.mapping[centLabel] *\
            ones(NTRIALS)
            for k, v in tmp.items():
                estimates[temp, :, ni + 1, model.mapping[k]] = v * ones((NTRIALS))
#print(percentages)
# plot max only

#fig, ax = subplots(2, sharex = 'all')
#
#m = estimates.max(-1)
#n = targets.max(-1)
#rcParams['axes.labelpad'] = 5
#for temp in range(NTEMPS):
#    for ni in range(2):
#        ax[0].scatter(n[temp, :, 0],  m[temp, :, ni])
#        ax[1].scatter(n[temp, :, 1], m[temp, :, ni])
#xlabel('causal')

tmp = zeros((NTEMPS, N + 1, COND))
fig, ax = subplots(1, 2, sharex = 'all', sharey = 'all')
rcParams['axes.labelpad'] = 50
mainax = fig.add_subplot(111, frameon = False, \
                         xticks = [],\
                         yticks = [],\
                         ylabel = 'Predicition accuracy(%)')
#mainax.set(ylabel = "Prediction accuracy")
subplots_adjust(wspace = 0)
width = .4
xloc  = arange(NTEMPS) * 2
conditions = [f'$M_r$ = {round(x, 2)}\n{y}' for x in temps for y in nudges]
condLabels = 'Underwhelming Overwhelming'.split()
labels = f'{information_impact}\tdeg\tclose\tbet\tev'.split('\t')

maxT = argmax(targets,  -1)

lll = f'max |{information_impact}|\t'
ii = '\t'.join(f'max |{i}|' for i in centralities)
lll += ii
lll = lll.split('\t')
for cond in range(COND):
    tax = ax[cond]
    tax.set_title(condLabels[cond])
    tax.set(xticks = xloc + .5 * width * N, \
            xticklabels = conditions,\
            )
    tax.tick_params(axis = 'x', rotation = 45)

    for temp in range(NTEMPS):
        for ni in range(N + 1):

            x = equal(maxestimates[temp, :, ni], maxT[temp, :, cond])
            y = x.mean() * 100
            x = xloc[temp]

            tmp[temp, ni, cond] =  x.mean()
            tax.bar(x + width * ni, y, width = width,\
              color = colors[ni], label = lll[ni])
            if y > 0:
                tax.text(x + width * ni, y + .5, \
                     int(y),\
                     fontweight = 'bold', \
                     horizontalalignment = 'center')
tax.legend(tax.get_legend_handles_labels()[1][:N + 1], loc = 'upper left' , bbox_to_anchor = (1,1),\
           borderaxespad = 0)
fig.savefig(figDir + 'statistics_overview.eps', format = 'eps', dpi = 1000)

# %% classfication with cross validation
from sklearn.feature_selection import SelectKBest, RFE, RFECV
from sklearn.svm import SVC
from sklearn import linear_model
from sklearn.feature_selection import chi2, mutual_info_classif
from sklearn.model_selection import StratifiedKFold, LeaveOneOut


from sklearn.ensemble import RandomForestClassifier
from sklearn import multioutput
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import accuracy_score

from sklearn.model_selection import GridSearchCV
import pandas as pd

# get continuous values
predLabels = ['ii', *centralities.keys()]
features     = estimates.max(-1).reshape(-1, N + 1)
# get their corresponding labels
featureClass = pd.DataFrame(\
            estimates.argmax(-1).reshape(-1, N + 1),\
            columns = predLabels)


# get correct labels
correctLabels = 'un ov'.split()
correct = targets.argmax(-1).reshape(-1, COND)
from sklearn.preprocessing import OneHotEncoder

import statsmodels.api as sm


y = pd.DataFrame(correct, columns = correctLabels)
Y = pd.DataFrame()
bias = ones((NTRIALS * NTEMPS, 1))

yy = zeros((NTEMPS * NTRIALS, N+1, COND))
for tidx, trueLabel in enumerate(y):
    for pidx, classPred in enumerate(featureClass):
        lab    = f'{trueLabel}_{classPred}'
        x = featureClass[classPred]
        ty= y[trueLabel] == x
        yy[:, pidx, tidx] = ty
        Y[lab] = ty

percentages = yy.reshape(NTEMPS, NTRIALS, N + 1, COND)

percentages = (percentages * (NTRIALS - 1) + .5 ) /(NTRIALS)
percentages = percentages.mean(1)

class LogitRegression(linear_model.LinearRegression):
    def fit(self, x, p):
        p = np.asarray(p)
        y = log(p) - log(1 - p)
        y[isfinite(y) == False] = 0
        return super().fit(x, y)

    def predict(self, x):
        y = super().predict(x)
        return 1 / (np.exp(-y) + 1)

bias = zeros((NTEMPS, 1))
X    = estimates.max(-1).mean(1)
#X    = hstack((bias, x))
ps =percentages[...,0]
clf = LogitRegression()
clf.fit(X, ps)

pred = clf.predict(X)
fig, ax = subplots()
for i in range(NTEMPS):
    ax.scatter(X[i], pred[i], label = temps[i])
ax.legend()
clf.fit(X[:, [0]], ps[:, 0])
ax.scatter(X[:, [0]], clf.predict(X[:, [0]]))

# %% 3d scatter per centrality?
N = len(centralities)
from mpl_toolkits.mplot3d import Axes3D
close('all')
gs = dict(\
          height_ratios = ones(NTEMPS) * 30,\
          width_ratios = [1,1])
gs = {}
fig, ax = subplots(3,2,\
                   subplot_kw = dict(\
                                     projection = '3d'),\
                    figsize = (30, 30),\
                    gridspec_kw = gs)

markers = array(['P', 'v','*',\
                         's', '*',\
                         'h', 'H',\
                         'D', 'd',\
                         'P', 'X'])
def removeLabels(ax, whichax, targets):
    """
    removes unwanted tick labels while remaining the grid
    """
    options = dict(x = ax.xaxis,\
                   y = ax.yaxis,\
                   z = ax.zaxis)
    tax    = options[whichax]
    labels = tax.get_ticklabels()
    locs   = tax.get_ticklocs()
    for idx, (label, val) in enumerate(zip(\
                           labels,\
                           locs\
                           )):
        if val not in targets:
            label.set_visible(False)
#    tax.set_ticks(locs)
    tax.set_data_interval(min(targets), max(targets))

#        if float(label.get_text()) not in targets:
#            label.set_visible(False)
#        if val not in targets: label.set_visible(False)


from matplotlib.patches import Patch
for ni in range(N):
    mark = markers[ni]
    for cond in range(COND):
        for t in range(NTEMPS):
            tax = ax[t, cond]

            tax.view_init(elev= 23, azim= 130)
            x, y   = estimates[t, :, [0, ni + 1]]
            sx = (x - x.min(1)[:, None]) / (x.max(1)[:, None] - x.min(1)[:, None])
            sy = (y - nanmin(y,1)[:, None]) / (nanmax(y, 1)[:, None] - nanmin(y,1)[:, None])
            z   = targets[t, :, cond, :]
            z   = (z - z.min(1)[:, None])/ (z.max(1)[:, None] - z.min(1)[:, None])
#            mark = markers[t]
            for node in range(NODES):
                tax.scatter(sx[:, node], sy[:, node], \
                            zs = z[:, node], \
                            color = colors[node],\
                            alpha = 1, s = 200, marker = mark)

            # format axes
            samesies = 35
            tax.set_xlabel(f'Information impact', fontsize = 40, \
                           labelpad = samesies)
            tax.set_ylabel('Centrality',fontsize = 40, \
                           labelpad = samesies)
            tax.set_zlabel(f'Causal impact', fontsize = 40, \
                           labelpad = samesies)
#            for axi in 'x y z'.split():
#                removeLabels(tax, axi, [0, 1])
#                tax.tick_params(axi, pad = 10, labelsize = 25)
            # hack the lmits
            if cond == 0:
                tax.text(1, 0, 1,\
                     f'$M_r$ = {round(temps[t], 2)}', \
                     fontdict = dict(fontsize = 40),\
                     transform = tax.transAxes,\
                     horizontalalignment = 'right', zorder = 5)
            if t == 0 :
                tax.set_title(condLabels[cond], y = 1.1, \
                              fontsize = 50)

[tax.invert_yaxis() for tax in ax.ravel()]

# dummy legend
samesies = 35
elements = [
        Line2D([0], [0], color = colors[i],\
              label = model.rmapping[i],\
              marker = 'o',\
              linestyle = '', markersize = samesies) for i in range(NODES)]

mainax = fig.add_subplot(111, \
                         frameon = False,\
                         xticks = [],\
                         yticks  = [])
leg1 = mainax.legend(handles = elements, \
                     bbox_to_anchor = (0, -0.05),\
                     loc = 'upper left',\
                     borderaxespad = 0, \
                     fontsize = 40, ncol = NODES // 3, \
                     title = 'Node', \
                     title_fontsize = 30, \
                     frameon = False,\
                     handletextpad=0.05, \
                     labelspacing = .1, \
                     handlelength = .9)

elements = [
        Line2D([0], [0], color = 'k',\
              label = centLabels[i], marker = markers[i],\
              linestyle = '', markersize = samesies) for i in range(N)]

leg2 = legend(handles = elements, \
                     bbox_to_anchor = (0.75, -0.05),\
                     loc = 'upper left', \
                     borderaxespad = 0, \
                     fontsize = 40, \
                     title = 'Marker',\
                     title_fontsize = 30,
                     ncol = N // 2,\
                     frameon = False)
[mainax.add_artist(e) for e in [leg1, leg2]]
fig.subplots_adjust(hspace = .2, wspace = .2)
fig.savefig(figDir + '3dscattercent.eps')
#        [tax.tick_params(axis = i, pad = 40) for i in 'x y z'.split()]

#%% randomforrest
from sklearn import model_selection
groups = arange(N + 1)
cv = model_selection.LeaveOneOut()

ty = yy[..., 0]
ty = correct[:, 0]
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor




scores = zeros((\
                ty.shape[0], \
                COND))
shuffescores = zeros(scores.shape)
imF = zeros((\
             ty.shape[0], \
             N + 1,\
             COND))

shuffscores = zeros(\
                    (\
                    features.shape[0],\
                    features.shape[1],\
                    COND,
                    )\
                     )

#for i, n in enumerate(nestims):
#    clf.n_estimators = n
#    for j, depth in enumerate(treedepths):
#        clf.max_depth = depth

# The list of hyper-parameters we want to optimize. For each one we define the bounds,
# the corresponding scikit-learn parameter name, as well as how to sample values
# from that dimension (`'log-uniform'` for the learning rate)
NOBS   = NTRIALS * NTEMPS


clf = RandomForestClassifier(\
                            n_estimators = 100,\
                             n_jobs = -1,\
                             criterion = 'entropy',\
                             bootstrap = True,\
                             )
from sklearn.tree import export_graphviz
for cond in range(COND):
    ty = yy[...,cond]
    for k, (train, test) in enumerate(cv.split(ty)):
        xi, xj = features[train], features[test]
        yi, yj = ty[train], ty[test]

        clf.fit(xi, yi)

        pr = clf.predict(xj)
        s = metrics.accuracy_score(yj, pr)
        scores[k, cond] = s
        imF[k, :, cond] = clf.feature_importances_

        for shuf in range(features.shape[1]):
            tf = features.copy()
            random.shuffle(tf[:, shuf])
            shuffpred = clf.predict(tf[test])

            s = metrics.accuracy_score(yj, shuffpred)
            shuffscores[k, shuf, cond] = s
# %% plot score and feature importance
rcParams['axes.labelpad'] = 5
fig, ax = subplots(3,2, sharex = 'row', sharey = 'row',\
                   figsize = (10,15))
x = arange(N + 1) * (1 + width) * 2
width = .5
mimF = imF.reshape(NTEMPS, NTRIALS, N + 1, COND).mean(1)
simF = imF.reshape(NTEMPS, NTRIALS, N + 1, COND).std(1)
s    = scores.reshape(NTEMPS, NTRIALS, COND)
ss   = scores.reshape(NTEMPS, NTRIALS, COND).mean(1)
sss  = scores.reshape(NTEMPS, NTRIALS, COND).std(1)
t    = arange(NTEMPS)

sc   = shuffscores.reshape(NTEMPS, NTRIALS, N+1, COND).mean(1)


d = (scores[:, None] - shuffscores) # / (scores)
d = d.reshape(NTEMPS, NTRIALS, N + 1, COND)
d = d.mean(1) / s.mean(1)[:, None] * 100
for temp in range(NTEMPS):
    for cond in range(COND):
        if temp == 0:
            ax[0, cond].set_title(condLabels[cond], fontsize = 30)
            
        tax = ax[0, cond]
        if temp == 0:
            tax.axhline(.5, color = 'black', linestyle = 'dashed')
        
            
        
        tax.bar(t[temp], ss[temp, cond], width = width, \
                color = colors[0])
        
#        tax.boxplot(s[temp, :, cond], \
#                )
        tax = ax[1, cond]
        tax.bar(x + width * temp, mimF[temp, :, cond],\
                width = width, label = f'$M_r$ ={round(temps[temp], 2)}')
        tax.axhline(1/5, color = 'black', linestyle = 'dashed')

        tax = ax[2, cond]
        tax.bar(x + width * temp, d[temp, ..., cond], \
                width = width)
#ax[0, 0].boxplot(s[..., 0].T)
ax[0,0].set(xticklabels = temps,\
            xticks = t, \
           )

tmpax = fig.add_subplot(311, frameon = 0, xticks = [], yticks = [])
tmpax.set_xlabel('$M_r$', fontsize = 20, labelpad = 25)
ax[0, 0].set_ylabel('Accuracy score', fontsize = 25)

height = 1.05
ax[0, 0].plot(t, height * ones(t.size) + .025, '-k')
ax[0, 0].text(t[1],  height + .015, '*', fontsize = 20)
ax[0, 0].set(ylim = (0, 1.3))

ax[0, 1].plot(t, height * ones(t.size) + .025, '-k')
ax[0, 1].text(t[1],  height + .015, '*', fontsize = 20)
ax[0, 1].set(ylim = (0, 1.3))

ax[1, 0].plot(x, height * ones(x.size) + .025, '-k')
ax[1, 0].text(x[x.size // 2],  height + .015, '*', fontsize = 20)
ax[1, 0].set(ylim = (0, 1.3))

ax[1, 1].plot(x, height * ones(x.size) + .025, '-k')
ax[1, 1].text(x[x.size // 2],  height + .015, '*', fontsize = 20)
ax[1, 1].set(ylim = (0, 1.3))

#ax[2, 0].plot(x, 100 * ones(x.size) + .025, '-k')
#ax[2, 0].text(x[x.size // 2],  100 + .015, '*', fontsize = 20)
#ax[2, 0].set(ylim = (0, 115))

centralities = {
                    r'$c_i^{deg}$' : partial(nx.degree, weight = 'weight'), \
                    r'$c_i^{betw}$': partial(nx.betweenness_centrality, weight = 'weight'),\
                    r'$c_i^{ic}$'  : partial(nx.information_centrality, weight = 'weight'),\
                    r'$c_i^{ev}$'  : partial(nx.eigenvector_centrality, weight = 'weight'),\
             }
ax[1,0].set(xticks = x +  width * 3/NTEMPS, \
            xticklabels = '',\
            )

ax[1, 0].set_ylabel('Feature importance', fontsize = 25)
ax[1,1].legend(fontsize = 25)

ax[2, 0].set_ylabel('delta score(%)', fontsize = 25)
ax[2, 0].set(
             xticks = x +  width * 3/NTEMPS, \
             xticklabels = ['$\mu_{max}$', *(i.replace('_i', '_{max}') for i in centralities.keys())],\
             )
idx = 35
ax[2,0].tick_params('x', labelsize = idx, rotation = 60)
ax[2,1].tick_params('x', labelsize = idx, rotation = 60)
fig.subplots_adjust(wspace = 0)
fig.savefig(figDir + 'classifier_stats.eps')

# %% rnf classifier simplified
fig, ax = subplots()

tmp = np.arange(COND)

ax.bar(tmp, scores.mean(0), width = .5)
ax.axhline(.5, linestyle = 'dashed', color = 'k')
ax.set_ylabel('Prediction accuracy')

ax.set_xticks(tmp)
ax.set_xticklabels('Underwhelming Overwhelming'.split())
ax.annotate('*', xy = (0, 1.1), fontsize = 50)
ax.annotate('*', xy = (1, 1.1), fontsize = 50)
ax.set_ylim(0, 1.3)



fig.savefig(figDir + 'accuracy_single.png')
fig, ax = subplots()
tmp = np.arange(5) - .25
for i in range(COND):
    j = labels[i]
    tax = ax
    tax.bar(tmp + i * .25, mimF.mean(0)[..., i], width = .5, label = j)
    tax.set_xticks(tmp + .25)
ax.axhline(1/5, color = 'k', linestyle = 'dashed')
#ax.plot(tmp + .25, ones(tmp.size) * 1.1, color = 'k')
#ax.annotate('*', (tmp[tmp.size // 2], 1.1), fontsize = 50)
#ax.set_ylim(0, 1.3)
tax.set_ylabel('Feature importance')
tax.legend()

tax.set_xticklabels(['Information\nimpact', *(f.func.__name__.split('_')[0] for f in centralities.values())], \
                     fontsize = 30, rotation = 60)
fig.savefig(figDir + 'feature_single.png')
    
# %%
from matplotlib import gridspec
fig = figure()

gs1 = gridspec.GridSpec(1,1)
gs1.update(left = 0.05, right = 0.48, wspace = 0.05)
left = fig.add_subplot(gs1[0]) # ((2,3), (0,0), colspan = 1, rowspan = 2, fig = fig)

gs2 = gridspec.GridSpec(3, 2)
gs2.update(left = .55, right = .98, hspace = 0, wspace = 0)
for idx, (i, j, tmp) in enumerate(zip(d.mean(0), mimF.mean(0), list(gs2)[1:])):
    ax = fig.add_subplot(tmp)
    # if ax.set(yticklabels = [], xticklabels = [])
    ax.scatter(i,j)
# left.set_title('Prediction accuracy')
# left.bar([0,1], ss.mean(0))
   
# left.set_ylabel('Accuracy score')
# left.set(xticks = [0, 1], xticklabels = conditionLabels)
# left.axhline(.5, color = 'red', linestyle = 'dashed', label = 'random choice')
# [left.annotate('*', (i, j + .005), fontsize = 20) for i, j in zip([0, 1], ss.mean(0))]

# xr = arange(len(labels))
# left.legend(loc = 'upper left')

# labels = ['$\mu_i$', *centLabels]
# for idx, label in enumerate(labels):
#     r, c  = unravel_index(idx, (2,2))
#     gs = gridspec.GridSpec
#     ax = subplot2grid((2, 3), (r, c + 1), fig = fig)
    # ax.scatter(d.mean(0)[idx], mimF.mean(0)[idx])
# right.set(xlabel = '$\delta$ score(%)', ylabel = 'Feature importance')
# [right.bar(xr + width * idx, i, width = width) for idx, i in enumerate(mimF.mean(0).T)]

# %%
d = array([(s - j) / s for s, j in zip(scores, shuffescores)])


#s = (scores - shuffscores) / (

p = scipy.stats.binom_test(scores.sum(0)[1], 60, .5)

O    = imF.sum(0)
E    = NOBS * 1 // (N + 1) * ones(O.shape, dtype = int)
test, pp = scipy.stats.chisquare(O, E, ddof = O.shape[1] - 1, axis = 0)

cO = O[0]
iO = O[1:].mean(0)

OO = zeros((2, COND))
OO[0, :] = cO
OO[1, :] = iO
EE = NOBS // OO.shape[0] * ones(OO.shape)

ptest, postp = scipy.stats.chisquare(OO, EE, axis = 0)
print(ptest, postp)

# correct for number of tests (n conditions + 1 post)
pp /= N + 2
postp /= N + 2
print(pp, postp)
# %% write results
import xlsxwriter
f = f'{extractThis}_statdata.xlsx'
writer = pd.ExcelWriter(f, engine = 'xlsxwriter')

pscores = pd.DataFrame(scores, columns = 'Underwhelming Overwhelming'.split())
pscores.to_excel(writer, sheet_name = 'RNF_scores')

# squash conditions
pshuffscoresmean = shuffscores.mean(0)
pshuffscoresstd  = shuffscores.std(0) * 2
pshuffscores      = stack((pshuffscoresmean, \
                            pshuffscoresstd), axis = 0)

pimF    = imF.mean(0)
pimFstd = imF.std(0)
pimFs   = stack((pimF , pimFstd), axis = 0)

# write to xlsx
for cond in range(COND):
    pshuffscore      = pd.DataFrame(\
                                pshuffscores[..., cond],\
                                columns = ['information impact',\
                                           *centLabels],\
                               index =  ['mean', 'std'])
    pshuffscore.to_excel(writer, sheet_name = f'shscores{condLabels[cond]}')

    pimF = pd.DataFrame(pimFs[..., cond], \
                     columns = ['information impact',\
                                           *centLabels],\
                               index =  ['mean', 'std'])
    pimF.to_excel(writer, sheet_name = f'fi_{condLabels[cond]}')
writer.save()
#    pd.MultiIndex(tuples, names = 'observed methodd )
#for cond in range(COND):


# %%
rcParams['axes.labelpad'] = 5

import pandas as pd

# intervention - scores


meanF  = imF.mean(0)
meanSS = shuffscores.mean(0)

t = {}

for cond, cl in enumerate(condLabels):
    t[cl] = t.get(cl, {})
    for idx, name in enumerate(labels):
        tmp = {\
                   'Feature importance' : meanF[idx, cond],\
                   'Shuffle score'      : meanSS\
               }
        t[cl][name] = t[cl].get(name, tmp)
t = dict(Method = t)
print(pd.DataFrame.from_dict(t))


#%%
fig, ax = subplots()

ac = Y.sum(0)[:, None]
counts = ac @ ac.T
showThis = sqrt(counts) / Y.shape[0]
h  = ax.imshow(cov(Y.T))
plotz.colorbar(h)
yl = [i for i in Y.columns]
yr = arange(len(yl))
ax.set(\
       xticks = yr,\
       yticks = yr,\
       xticklabels = yl, \
       yticklabels = yl,\
       title = 'Covariance of correct predictions')
ax.tick_params(axis = 'x', rotation = 45)
fig.savefig(figDir + 'cov_predictions.eps')

fig, ax = subplots()

mainax = fig.add_subplot(121, frameon = 0,\
                         xticks = [],\
                         yticks = [])
mainax.set_xlabel('Underwhelming', labelpad = 100)
mainax = fig.add_subplot(122, frameon = 0,\
                         xticks = [],\
                         yticks = [])
mainax.set_xlabel('Overwhelming', labelpad = 100)
x = arange(Y.shape[1])
ax.bar(x, Y.mean(0))
ax.set(xticks = yr, xticklabels = yl, ylabel = 'Prediction accuracy(%)')
ax.tick_params(axis = 'x', rotation = 45)
#%%



#from sklearn.ensemble import RandomForestClassifier, \
#

#subplots_adjust(wspace = 0)
#subplots_adjust(hspace = .5)
#allpredset = set(array(featureClass).flatten())
#alltrueset = set(array(y).flatten())
#mapper = {lab : jdx for jdx, lab in enumerate(\
#                          alltrueset.union(allpredset))}
#fig, ax = subplots(1, 1)
##mainax  = fig.add_subplot(111,\
##                              frameon = False,\
##                              xticks  = [],\
##                              yticks  = [],\
##                              ylabel  = f'{fc}\n\n')
#for n, fc in enumerate(featureClass):
#
#    nn = len(mapper)
#    conf = zeros((nn, nn))
#    tax = ax
#    for cond, tc in enumerate(y):
#        for true, pred in zip(y[tc], featureClass[fc]):
#            xi = mapper[true]
#            yi = mapper[pred]
#            c = 0
#            if true and pred:
#                c = 1
#            conf[xi, yi] += c
#    conf = (conf.ravel() / y.shape[0]).reshape(nn,nn)
#    conf[isfinite(conf) == False] = 0
#    print(conf)
#
#    h = tax.imshow(conf, cmap = cm.plasma,\
#                   vmin = 0,\
#                   vmax = 1)
#    tax.set(\
#        yticklabels = mapper, \
#        xticklabels = mapper, \
#        xticks      = arange(nn),\
#        yticks      = arange(nn),\
#        )
#    tax.set_xlabel('Predicted', labelpad = 5)
#    tax.set(title = tc + '\n')



# %% kmeans plots

from sklearn import cluster
from sklearn import mixture

nclus = arange(2, model.nNodes - 1)

tmp = estimates[...,0, :].reshape(-1, 1)
y   = moveaxis(targets, -2, -1).reshape(-1, COND)
x   = hstack((tmp, y)).reshape(NTEMPS, NTRIALS * NODES, COND + 1)

from sklearn.metrics import silhouette_samples, silhouette_score


gs = dict(\
          height_ratios = ones(NODES  - nclus.min()-1),\
          width_ratios  = [2, 1])
from sklearn.preprocessing import StandardScaler

scores_ = zeros((NTEMPS, nclus.size, COND))
for temp in range(NTEMPS):
    for cond in range(COND):
        fig, ax = subplots(NODES  - nclus.min() - 1, 2, \
                           sharex = 'col', sharey = 'col', \
                           gridspec_kw = gs, \
                           figsize = (5, 15))

        mainax = fig.add_subplot(111, xticks = [],\
                                 yticks = [],\
                                 frameon = False)
        mainax.set_title(f'$M_r$ = {temps[temp]}\n')
    #    mainax = fig.add_subplot(121, xticks = [],\
    #                             yticks = [],\
    #                             frameon = False, \
    #                             )
        mainax.set_ylabel(f'Z-scored {causal_impact}', labelpad = 40)


        subplots_adjust(wspace = 0)
        for idx, n in enumerate(nclus):
            tax = ax[idx, 1]
            clf = cluster.KMeans(n)
    #        clf = mixture.BayesianGaussianMixture(n, covariance_type = 'full')
            xi = x[temp, :, [0, cond + 1]].T
            xi = StandardScaler().fit_transform(xi)
            ypred = clf.fit_predict(xi)
            sil = silhouette_score(xi, ypred)
            sils= silhouette_samples(xi, ypred)

            scores_[temp, idx, cond] = sil
            tax.set_title(f'N={n}')
            low = 2
            for ni in range(n):
                tmp_s = sils[ypred == ni]
                tmp_s.sort()
                high = low + tmp_s.size

                color = cm.tab20(ni / n)
                color = colors[ni]
                tax.fill_betweenx(arange(low, high), \
                                  0, tmp_s, color = color,)
                low = high + 2
    #        tax.set(yscale = 'log')
            tax.axvline(sil, color = 'red', linestyle = '--')
            tax.set(yticks = [])
            tax = ax[idx, 0]
            tax.scatter(*xi.T, c = colors[ypred])
        ax[-1, 0].set_xlabel('Z-scored $\mu_i$', labelpad = 15)
        ax[-1, 1].set_xlabel('Silhouette score', labelpad = 15)
        fig.savefig(figDir + f'$M_r$ = {round(temps[temp], 2)}_kmeans{cond}.eps')
# %% optimal kmeans plot

fig, ax = subplots(1,2, sharey = 'all')
rcParams['axes.labelpad'] = 60

sfig, sax = subplots(3,2, sharey = 'row')

maxes = zeros((NTEMPS, COND))

y   = moveaxis(targets, -2, -1).reshape(-1, COND)
x   = hstack((tmp, y)).reshape(NTEMPS, NTRIALS * NODES, COND + 1)

for i in range(COND):
    tax = ax[i]
    s = scores_[...,i]
    for t in range(NTEMPS):

        tax.plot(nclus, s[t], label = f'{round(temps[t],2)}', \
                 color = colors[t])

        m = argmax(s[t])
        maxes[t, i] = m
        tax.axvline(x = nclus[m], color = colors[t], \
                    linestyle = '--')

        # show cluster
        xi = x[t, :, [0, i + 1]].T # get correct data
        xi = StandardScaler().fit_transform(xi) # zscore
        clf = cluster.KMeans(nclus[m]) # kmeans
        ypred = clf.fit_predict(xi).reshape(NTRIALS, NODES)
        xi    =xi.reshape(NTRIALS, NODES, 2)
        ttax  = sax[t, i]

        markers = array(['o', 'v', '^',\
                         '<', '>', '8',\
                         's', 'p', '*',\
                         'h', 'H',\
                         'D', 'd',\
                         'P', 'X'])
        for node in range(NODES):
            for trial in range(NTRIALS):
                ttax.scatter(*xi[trial, node], color = colors[node],\
                         marker = markers[ypred[trial, node]], \
                         )

        # format axes
        if t == 0:
            ttax.set_title(condLabels[i])

        if i == 0:
            ttax.text(.25 if i == 0 else .85, .85 , \
             f'$M_r$ = {round(temps[t], 2)}', \
             fontdict = dict(fontsize = 15),\
             transform = ttax.transAxes,\
             horizontalalignment = 'right')

    tax.set_title(condLabels[i])


tax.legend(\
           loc = 'bottom right', \
           title = 'Temperature', \
           title_fontsize = 15,\
           )
mainax = fig.add_subplot(111,\
                         frameon = False,\
                         xticks = [],\
                         yticks = [])


mainax.set(xlabel = 'k clusters', \
            ylabel = 'silhouette score (s)')


fig.subplots_adjust(wspace = 0)
sfig.subplots_adjust(wspace = 0 , hspace = 0)
mainax = sfig.add_subplot(111,\
                         frameon = False,\
                         xticks = [],\
                         yticks = [])
mainax.set(xlabel = f'Z-scored information impact {information_impact}',\
           ylabel = f'Z-scored Causal impact {causal_impact}')

fig.savefig(figDir + 'mean_silscores.eps')
elements = [\
            Line2D([0],[0], \
                   color = colors[i], \
                   label = model.rmapping[i],\
                   marker = 'o', linestyle = '',\
                   markersize = 15) \
           for i in range(NODES)]

mainax.legend(handles = elements, loc = 'upper left',\
              bbox_to_anchor = (1,1),
              frameon = False, borderaxespad = 0)
sfig.savefig(figDir + 'optimal_clusters.eps')

#%% multiple linear regression
import statsmodels.api as sm
tmp = ['intercept', \
       '$\mu_i$',\
       *centLabels]
def imshow_symlog(ax, my_matrix, vmin, vmax, logthresh=5):
    img= ax.imshow( my_matrix ,
                vmin=float(vmin), vmax=float(vmax),
                norm=matplotlib.colors.SymLogNorm(10**-logthresh) )
    return img
X = moveaxis(estimates, 2, -1)
y = moveaxis(targets, 2, -1).reshape(-1, COND)
y = scipy.stats.zscore(y, 0)
#X = scipy.stats.zscore(X, 2)
X = X.reshape(-1, N + 1)
X = scipy.stats.zscore(X, 0)
X[isfinite(X) == False] = 0
y = pd.DataFrame(y, columns = 'Underwhelming Overwhelming'.split())


#X = X[:, [0]]

# visualize covariance
fig, ax =  subplots()
#h = imshow_symlog(ax, abs(corrcoef(array(X)[:, 1:].T)), 0, 1)
cvar = corrcoef(array(X).T)
h = ax.imshow(cvar, vmin = -1, vmax = 1)
ax.set(xticks = arange(N+1),\
       yticks = arange(N+1),\
       xticklabels = tmp[1:], \
       yticklabels = tmp[1:])
cbar = plotz.colorbar(h)
cbar.set_label('R', rotation = 0)
fig.savefig(figDir + 'covariance.eps')
# fit model
X = sm.add_constant(X)

X = pd.DataFrame(X, columns = tmp[:X.shape[1]])

est = sm.OLS(y['Underwhelming'], X).fit()

mr = est.summary().as_latex()
print(est.summary())
import csv
with open(f'{extractThis}.linregress.tex', 'w') as f:
    f.write(mr)


# %%
# % fit the results
rcParams['axes.labelpad'] = 5
linf = lambda x, beta, c : beta * x + c
bbox_props = dict(fc="white", lw=2)
mins, maxs = X.min(0), X.max(0)
fig, ax = subplots(2, 3, sharex = 'all', sharey = 'all')

#fig = figure(2, 3)
import matplotlib.patheffects as pe
# skip intercept
elements = []
for idx, i in enumerate(X.columns[1:]):
#    tax = ax[idx]
    tax = ax.ravel()[idx]
    if idx == N:
        tax = ax.ravel()[idx + 1]
        ax.ravel()[idx].axis('off')
#    row = 0 if idx < 3 else 1
#    col = idx * 2 if idx < 3 else row

#    print(row, col)
#    tax = subplot2grid((2, 6), loc = (row, col), colspan = 2)
    x = linspace(mins[i], maxs[i])
    tax.scatter(X[i], y['Underwhelming'],  alpha = 1, \
               color = colors[idx], label = i)
    beta = est.params[i]
    
    ex, ey = est.bse['intercept'], est.bse[i]
    tax.plot(x, linf(x, beta, est.params['intercept']), color = colors[idx],\
            linestyle = 'dashed', path_effects=[pe.Stroke(linewidth=5, foreground='k')])
    
    tax.fill_between(x, linf(x, beta + 2 * ex, est.params['intercept'] + 2 * ex), \
                     linf(x, beta - 2 * ex, est.params['intercept'] - 2 * ex),\
                     color = 'k', alpha = 0.5)
    elements.append(Line2D([0], [0], color = colors[idx], marker = '.', \
                           linestyle = 'none', label = i, markersize = 40))
    ax[0,0].text(.9, .9, f"$R^2$ = {est.rsquared: .2f}", horizontalalignment = 'right',\
         transform = tax.axes.transAxes, fontsize = 20, bbox = dict(\
                                                                          fc = 'white'))
    if est.pvalues[i] < .05:
        tmp = x[x.size // 2]
        xy  = (tmp, linf(tmp, beta, est.params['intercept']))
        bbox_props['ec'] = colors[idx]
        theta = 45 * beta# arcsin(beta)/ (2 * pi)* 180,
        t = tax.text(*xy, fr"$\beta$={round(beta, 2):.1e}", rotation = theta,\
                                      fontsize = 20,\
                                      bbox=bbox_props)
   
        bb = t.get_bbox_patch()
#        bb.set_boxstyle('rarrow', pad = .5)
#        tax.annotate(fr"$\beta$={round(beta, 2):e}", xy = xy, \
#                 xytext = xy, xycoords = 'data', \
#                 textcoords = 'data', rotation = 45, fontsize =  20)


mainax = fig.add_subplot(111, frameon = False, \
                         xticks = [], \
                         yticks = [], \
                         )
ax[1, 1].legend(handles = elements, loc = 'lower center', title = 'x', title_fontsize = 35,\
              bbox_to_anchor = (.5, -.1), fontsize = 23, frameon = False, \
              handletextpad = 0)


mainax.set_xlabel('Z-scored x', labelpad = 40, fontsize = 30)
mainax.set_ylabel(f'Z-scored causal impact ({causal_impact})', labelpad = 40, fontsize = 30)
fig.subplots_adjust(hspace = 0, wspace = 0)

#tax.legend(title = 'x', title_fontsize = 15)



#tax.set(xlabel = 'Z-scored x',\
#        ylabel = f'Z-scored {causal_impact}')
fig.savefig(figDir + 'multiregr.eps')

#assert 0

# %% appendix box rejection
rcParams['axes.labelpad'] = 40
fig, ax = subplots(1, COND, sharey = 'all')
mainax = fig.add_subplot(111, frameon = False,\
                         xticks = [],\
                         yticks = [],\
                         ylabel = 'Rejection rate(%)'\
                         )

subplots_adjust(wspace = 0)
x     = arange(NTEMPS) * 3

width = .2
labels = 'Underwhelming Overwhelming'.split()
conditions = [f'$M_r$ = {round(i,2)}' for i in temps]
for cond in range(COND):
    tax = ax[cond]
    tax.set_title(labels[cond])
    tax.set(xticks = x + .5 * width * NODES, \
            xticklabels = conditions,\
            )
    tax.tick_params(axis = 'x', rotation = 45)

    for node in range(NODES):
        y  = rejections[cond, :, node] * 100
        tx = x + width * node
        tax.bar(tx, y, width = width,\
                color = colors[node], label = model.rmapping[node])
tax.legend(loc = 'upper left',\
          bbox_to_anchor = (1, 1),\
          borderaxespad = 0, \
          frameon = False)
fig.savefig(figDir + 'rejection_rate.eps', format = 'eps', dpi = 1000)

## %% appendix plot: fit error comparison
fig, ax = subplots()
errors = errors.reshape(-1, NODES)
subplots_adjust(hspace = 0)
width= .07
locs = arange(0, len(errors))
for node in range(NODES):
    ax.bar(locs + width * node, \
           errors[:, node], color = colors[node],\
           width = width, label = model.rmapping[node])

conditions = [f'$M_r$ = {round(x, 2)}\n{y}' for x in temps for y in nudges]
# set labels per group and center it
ax.set(yscale = 'log', xticklabels = conditions, \
       xticks = locs + .5 * NODES * width, ylabel = 'Mean square error')
ax.tick_params(axis = 'x', rotation = 45)
ax.legend(loc = 'upper left',\
          bbox_to_anchor = (1, 1),\
          borderaxespad = 0, \
          frameon = False)
rcParams['axes.labelpad'] = 1
fig.savefig(figDir + 'fiterror.eps', format = 'eps', dpi = 1000)


# %%
from scipy import spatial
a = np.argsort(aucs)[..., -2:]
b = []
dis = np.zeros(a.shape[:-1])
for i, j in  enumerate(a):
    for k, l in enumerate(j):
        for m, n in enumerate(l):
            x, y  = aucs[i, k, m, n]
            dis[i,k,m] = np.sqrt((x - y) **2)
fig, ax = subplots(figsize = (12, 15))
l = ['Information impact', *condLabels]
for idx, (i, j) in enumerate(zip(dis.mean(-1), dis.std(-1))):
    ax.errorbar(np.arange(NTEMPS), i, j, alpha = .5, label = l[idx], zorder = 5 - idx)
ax.set_xlabel('Magnetization fraction ($M_r$)', fontsize = 35, labelpad = 4)
ax.set_ylabel('Mean distance ($\sqrt{( first - second )^2}$)', fontsize = 25, labelpad = 4)
ax.set(xticks = np.arange(NTEMPS), xticklabels = temps)
ax.tick_params(axis='both', which='major', labelsize=25)
ax.legend(fontsize = 25)
fig.savefig(figDir + 'distance.png')
# h = ax.imshow(dis.mean(-1))
# plotz.colorbar(h)
            
# %%
import scipy
t = np.arange(0, 20, 0.1)
d = np.exp(-5)*np.power(5, t)/scipy.misc.factorial(t)
fig, ax = subplots()
ax.plot(t, d)
ax.set(ylabel = 'Dynamic impact', xlabel = 'Degree')