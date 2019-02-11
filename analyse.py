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

close('all')
style.use('seaborn-poster')
dataPath = f"{os.getcwd()}/Data/"
#dataPath = '/mnt/'
# convenience
kite   = '1548347769.6300871'
psycho = '1548025318.5751357'


extractThis      = IO.newest(dataPath)[-1]
extractThis      = psycho
#extractThis      = kite
#extractThis      = '1547303564.8185222'
#extractThis  = '1548338989.260526'
extractThis = extractThis.split('/')[-1] if extractThis.startswith('/') else extractThis
loadThis = extractThis if extractThis.startswith('/') else f"{dataPath}{extractThis}"
data     = IO.DataLoader(loadThis)


settings = IO.readSettings(loadThis)
deltas   = settings['deltas']
repeats  = settings['repeat']


temps    = [float(i.split('=')[-1]) for i in data.keys()]
nudges   = list(data[next(iter(data))].keys())
temp     = temps[0]

# Draw graph ; assumes the simulation is over 1 type of graph
from Models.fastIsing import Ising
control  = IO.loadData(data[f't={temp}']['control'][0]) # TEMP WORKAROUND
model    = Ising(control.graph)

NTRIALS  = len(data[f't={temp}']['control'])
NTEMPS   = len(data)
NNUDGE   = len(data[f't={temp}'])
NODES    = model.nNodes

DELTAS_, EXTRA = divmod(deltas, 2) #use half, also correct for border artefact
COND     = NNUDGE - 1
DELTAS = DELTAS_ - 1

pulseSizes = list(data[f't={temp}'].keys())

print(f'Listing temps: {temps}')
print(f'Listing nudges: {pulseSizes}')

figDir = f'../thesis/figures/{extractThis}'

# %% # show mag vs temperature
tmp = IO.loadPickle(f'{loadThis}/mags.pickle')
fig, ax = subplots()
# if noisy load fmag otherwise load mag
mag = tmp.get('fmag', tmp.get('mag'))

ax.scatter(tmp['temps'], mag, alpha = .2)
ax.scatter(tmp['temperatures'], tmp['magRange'] * mag.max(), \
           color = 'red', zorder = 2)

func = lambda x, a, b, c, d :  a / (1 + exp(b * (x - c))) + d # tanh(-a * x)* b + c
a, b = scipy.optimize.curve_fit(func, tmp['temps'], mag.squeeze(), maxfev = 10000)
x = linspace(min(tmp['temps']), max(tmp['temps']), 1000)
ax.plot(x, func(x, *a), '--k')
ax.set(xlabel = 'Temperature (T)', ylabel = '|<M>|')
rcParams['axes.labelpad'] = 10
fig.savefig(figDir + 'temp_mag.eps', format = 'eps', dpi = 1000)
# %% plot graph

centralities = dict(\
                    deg_w  = partial(nx.degree, weight = 'weight'), \
#                    close  = partial(nx.closeness_centrality, distance = 'weight'),\
                    bet    = partial(nx.betweenness_centrality, weight = 'weight'),\
                    ic     = partial(nx.information_centrality, weight = 'weight'),\
                    ev     = partial(nx.eigenvector_centrality, weight = 'weight'),\
#             cfl = partial(nx.current_flow_betweenness_centrality, weight = 'weight'),
             )

# %%
fig, ax  = subplots(figsize = (10, 10), frameon = False)
ax.set(xticks = [], yticks = [])
positions = nx.nx_agraph.graphviz_layout(model.graph, prog = 'neato', \
                                         )
#                                         root = sorted(dict(model.graph.degree()).items())[0][0])

positions = {node : tuple(i * 1 for i in pos) for node, pos in positions.items() }
p = dict(layout = dict(scale = 1),\
         annotate = dict(fontweight = 'extra bold'),\
         cicle = dict(radius = 10),\
         )
#p = {}
plotz.addGraphPretty(model.graph, ax, positions, \
                     mapping = model.mapping,\
                     **p,\
                     )


#nx.draw(model.graph, positions, ax = ax)
ax.set_aspect('equal', 'box')
ax.set(xticks = [], yticks = [])
ax.axis('off')
fig.show()
savefig(figDir + 'network.eps')
#%%
ss = dict(\
          height_ratios = [1, 1], width_ratios = [1, 1])
centLabels = 'Degree Betweenness Information Eigenvector'.split()
fig, ax = subplots(2, 2, gridspec_kw = ss)

#plotz.addGraphPretty(model.graph, ax[0, 0], \
#                     positions, mapping = model.rmapping,
#                     )
from matplotlib.patches import Circle
for idx, (cent, cf) in enumerate(centralities.items()):
    c = dict(cf(model.graph))
    s = array(list(c.values()))
    s = (s - s.min()) /(s.max() - s.min())
    tax = ax[:, :].ravel()[idx]
    tax.axis('off')
    tax.set_aspect('equal','box')
    tax.set_title(centLabels[idx])
    plotz.addGraphPretty(model.graph, tax, positions, \
                     mapping = model.mapping,\
                     **p,\
                     )
    for pidx, pp in enumerate(tax.get_children()):
        if isinstance(pp, Circle):
            pp.set(radius = s[pidx] * pp.radius * 2.4 )
#fig.subplots_adjust(hspace = 0, wspace = 0)
fig.savefig(figDir +  'graph_and_cent.eps')

#assert 0
# %%


# data matrix
#dd       = zeros((NTRIALS, NODES, DELTAS, COND))

# extract data for all nodes
information_impact = '$\mu_i$'
causal_impact      = '$\gamma_i$'
from tqdm import tqdm


# define color swaps
colors = cm.tab20(arange(NODES))
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
    fileName = fileNames[fidx]
    # do control
    data = frombuffer(var_dict.get('X')).reshape(var_dict['xShape'])
    node, temp, trial = unravel_index(fidx, var_dict.get('start'), order = 'F')
    # control data
    if '{}' in fileName:
        # load control; bias correct mi
        control = IO.loadData(fileName)
        graph   = control.graph
        # panzeri-treves correction
        mi   = control.mi
        bias = stats.panzeriTrevesCorrection(control.px,\
                                             control.conditional, \
                                             repeats)
        mi -= bias
        data[0, trial, temp]  = mi[:DELTAS - EXTRA, :].T
    # nudged data
    else:

        targetName = fileName.split('_')[-1] # extract relevant part
        jdx = [model.mapping[int(j)] if j.isdigit() else model.mapping[j]\
                             for key in model.mapping\
                             for j in re.findall(str(key), re.sub(':(.*?)\}', '', targetName))]
        jdx = jdx[0]
        useThis = fidx - node

        # load matching control
        control = IO.loadData(fileNames[useThis])
         # load nudge
        sample  = IO.loadData(fileName)
#        bias    = stats.panzeriTrevesCorrection(control.px,\
#                                        control.conditional,\
#                                        repeats)
#        control.px -= bias[..., None]
#
#        bias = stats.panzeriTrevesCorrection(sample.px,
#                                              sample.conditional,\
#                                              repeats)
#        sample.px -= bias[..., None]

#        impact  = stats.hellingerDistance(control.px, px)
        impact  = stats.KL(control.px, sample.px)
        # don't use +1 as the nudge has no effect at zero
        redIm = nanmean(impact[DELTAS + EXTRA + 2:], axis = -1).T
        # TODO: check if this works with tuples (not sure)
        data[(node - 1) // NODES + 1, trial, temp, jdx,  :] = redIm.squeeze().T

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
                       key = lambda x: x.split('/')[-1].split('_')[0],\
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

settings = IO.readSettings(loadThis)
repeats  = settings['repeat']

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
fig, ax = subplots(3, 4, sharex = 'col', sharey = 'row', gridspec_kw = gs)
mainax  = fig.add_subplot(111, frameon = 0)
mainax.set(xticks = [], yticks = [])

#mainax.set_title(f'\n\n').
mainax.set_xlabel('Time[step]', labelpad = 40)

x  = arange(DELTAS)
xx = linspace(0, 1 * DELTAS, 1000)
sidx = 2 # 1.96
labels = 'Control\tUnderwhelming\tOverwhelming'.split('\t')
_ii    = '$I(s_i^t ; S^t_0)$'

[i.axis('off') for i in ax[:, 1]]

rcParams['axes.labelpad'] = 80
ax[1, 2].set_ylabel("$D_{KL}(P || P')$", labelpad = 5)
ax[1, 0].set_ylabel(_ii, labelpad = 5)
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
        for node, idx in sorted(model.mapping.items(), key = lambda x : x[1]):

            # plot mean fit
            tax.plot(xx, func(xx, *meanCoeffs[idx]),\
                     color = colors[idx],\
                     alpha = 1, \
                     markeredgecolor = 'black',\
                     label =  node)
            # plot the raw data
            tax.errorbar(x, mus[idx],\
                         fmt = '.',\
                         yerr = sidx * sigmas[idx],\
                         markersize = 15, \
                         color = colors[idx],\
                         label = node) # mpl3 broke legends?

#            ax[cidx].set(xscale = 'log')


        tax.ticklabel_format(axis = 'y', style = 'sci',\
                         scilimits = (0, 4))

    tax.text(.95, .9, \
             f'T={round(temps[temp], 2)}', \
             fontdict = dict(fontsize = 15),\
             transform = tax.transAxes,\
             horizontalalignment = 'right')
#    tax.set(xlim = (-1.5, 30))

        #    ax[cidx].set(xscale = 'log', yscale = 'log'
# format plot

mainax.legend(\
              tax.lines, [i.get_label() for i in tax.lines],\
              title          = 'Node', \
              title_fontsize = 15, \
              loc            = 'upper left', \
              bbox_to_anchor = (1, 1),\
              borderaxespad  = 0, \
              frameon        = False,\
              )
subplots_adjust(hspace = 0, wspace = 0)
fig.show()
fig.savefig(figDir + 'mi_time.eps', format='eps', dpi=1000, pad_inches = 0,\
        bbox_inches = 'tight')

show()

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

aucs = aucs_raw.copy()
thresh = 3
clf = MinCovDet()

labels = 'Underwhelming\tOverwhelming'.split('\t')
rejections = zeros((COND, NTEMPS, NODES))
showOutliers = True
showOutliers = False
pval = .01
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
            try:
                tmp = aucs_raw[[0, nudge], temp, :, idx]
                clf.fit(tmp.T)
                Z = clf.mahalanobis(np.c_[xx.ravel(), yy.ravel()])
                if showOutliers:
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
            except:
                continue

#            tax.scatter(*tmp.mean(1), color = 'k', s = 50, zorder = 5)
            alpha = .1 if showOutliers else 1
            tax.scatter(*aucs[[0, nudge], temp, :, idx], \
                        color = colors[idx], \
                        label = node, alpha = alpha, \
                        linewidth = 1)

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
                           borderaxespad = 0,\
                           title_fontsize = 15)
                # force alpha
                for l in leg.legendHandles:
                    l._facecolors[:, -1] = 1
        if nudge == 1:
             tax.text(.05, .9, \
             f'T={round(temps[temp], 2)}', \
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


# plot info
rcParams['axes.labelpad'] = 20
for condition, condLabel in enumerate(conditionLabels):
    fig, ax = subplots(len(centralities), 3, sharex = 'all', sharey = 'row')
    mainax = fig.add_subplot(111, frameon = False, xticks = [], yticks = [])
    mainax.set_xlabel(f'Causal impact({causal_impact})', labelpad = 50)
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
                tax.set_title(f'T={round(temps[temp], 2)}')
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
conditions = [f'T={round(x, 2)}\n{y}' for x in temps for y in nudges]
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
            tax.set_xlabel(f'{information_impact}', fontsize = 40, \
                           labelpad = samesies)
            tax.set_ylabel('Centrality',fontsize = 40, \
                           labelpad = samesies)
            tax.set_zlabel(f'{causal_impact}', fontsize = 40, \
                           labelpad = samesies)
            for axi in 'x y z'.split():
                removeLabels(tax, axi, [0, 1])
                tax.tick_params(axi, pad = 10, labelsize = 25)
            # hack the lmits
            if cond == 0:
                tax.text(1, 0, 1,\
                     f'T={round(temps[t], 2)}', \
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
                   figsize = (10,10))
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
            ax[0, cond].set_title(condLabels[cond])
#            ax[0, cond].errorbar(\
#                          t, ss[:, cond], yerr = sss[:, cond],\
#                          linestyle = 'none',\
#                          color = 'k',\
#                          )

        tax = ax[0, cond]
        tax.bar(t[temp], ss[temp, cond], width = width, \
                color = colors[0])
#        tax.boxplot(s[temp, :, cond], \
#                )
        tax = ax[1, cond]
        tax.bar(x + width * temp, mimF[temp, :, cond],\
                width = width, label = f'T={round(temps[temp], 2)}')

        tax = ax[2, cond]
        tax.bar(x + width * temp, d[temp, ..., cond], \
                width = width)
#ax[0, 0].boxplot(s[..., 0].T)
ax[0,0].set(xticklabels = [f'T={round(temp, 2)}' for temp in temps],\
            xticks = t, \
            ylabel = 'Accuracy score')

ax[1,0].set(xticks = x +  width * 3/NTEMPS, \
            xticklabels = '',\
            ylabel = 'Feature importance')
ax[1,1].legend()
ax[2, 0].set(ylabel = 'delta score(%)',\
             xticks = x +  width * 3/NTEMPS, \
             xticklabels = ['$\mu_i$', *centralities.keys()],\
             )
fig.subplots_adjust(wspace = 0)
fig.savefig(figDir + 'classifier_stats.eps')
# %%
d = (scores.mean(0) - shuffscores.mean(0)) / (scores.mean(0))
#s = (scores - shuffscores) / (

p = scipy.stats.binom_test(scores.sum(0)[1], 60, .5, 'greater')

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
        mainax.set_title(f'T={temps[temp]}\n')
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
        ax[-1, 0].set_xlabel('Zscored $\mu_i$', labelpad = 15)
        ax[-1, 1].set_xlabel('Silouette score', labelpad = 15)
        fig.savefig(figDir + f'T={round(temps[temp], 2)}_kmeans{cond}.eps')
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
             f'T={round(temps[t], 2)}', \
             fontdict = dict(fontsize = 15),\
             transform = ttax.transAxes,\
             horizontalalignment = 'right')

    tax.set_title(condLabels[i])


tax.legend(bbox_to_anchor = (.4, -0.05), borderaxespad = 0, \
               loc = 'upper left', frameon = False,\
               title = 'Temperature', \
               title_fontsize = 15,\
               ncol = 3)
mainax = fig.add_subplot(111,\
                         frameon = False,\
                         xticks = [],\
                         yticks = [])


mainax.set(xlabel = 'k clusters', \
            ylabel = 'sillouette score (s)')


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
linf = lambda x, beta : beta * x + est.params['intercept']
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
    tax.plot(x, linf(x, beta), color = colors[idx],\
            linestyle = 'dashed', path_effects=[pe.Stroke(linewidth=5, foreground='k')])
    elements.append(Line2D([0], [0], color = colors[idx], marker = '.', \
                           linestyle = 'none', label = i))
    if est.pvalues[i] < .05:
        tmp = x[x.size // 2]
        xy  = (tmp, linf(tmp, beta))
        bbox_props['ec'] = colors[idx]
        theta = 45 * beta# arcsin(beta)/ (2 * pi)* 180,
        t = tax.text(*xy, fr"$\beta$={round(beta, 2):.1e}", rotation = theta,\
                                      fontsize = 20,\
                                      bbox=bbox_props)
        tax.text(.9, .9, f"$R^2$ = {est.rsquared: .2f}", horizontalalignment = 'right',\
         transform = tax.axes.transAxes, fontsize = 20, bbox = dict(\
                                                                          fc = 'white'))
        bb = t.get_bbox_patch()
#        bb.set_boxstyle('rarrow', pad = .5)
#        tax.annotate(fr"$\beta$={round(beta, 2):e}", xy = xy, \
#                 xytext = xy, xycoords = 'data', \
#                 textcoords = 'data', rotation = 45, fontsize =  20)
        
        
mainax = fig.add_subplot(111, frameon = False, \
                         xticks = [], \
                         yticks = [], \
                         )
mainax.legend(handles = elements, loc = 'upper center', title = 'x', title_fontsize = 15,\
              bbox_to_anchor = (.5, .3))


mainax.set_xlabel('Z-scored x', labelpad = 40)
mainax.set_ylabel(f'Z-scored {causal_impact}', labelpad = 40)
fig.subplots_adjust(hspace = 0, wspace = 0)

#tax.legend(title = 'x', title_fontsize = 15)



#tax.set(xlabel = 'Z-scored x',\
#        ylabel = f'Z-scored {causal_impact}')
fig.savefig(figDir + 'multiregr.eps')



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
conditions = [f'T={round(i,2)}' for i in temps]
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

conditions = [f'T={round(x, 2)}\n{y}' for x in temps for y in nudges]
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
