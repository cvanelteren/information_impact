#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 18 12:09:36 2018

@author: casper
"""
from numpy import *
from scipy import optimize, integrate
from matplotlib.pyplot import *
from time import sleep
from Utils import plotting as plotz, stats, IO
import os, re, networkx as nx, scipy, multiprocessing as mp

close('all')
style.use('seaborn-poster')
dataPath = f"{os.getcwd()}/Data/"
#dataPath = '/mnt/'
extractThis      = IO.newest(dataPath)[-1]
#extractThis      = '1548025318.5751357' # psycho
#extractThis      = '1548347769.6300871' # kite neg
#extractThis      = '1547303564.8185222'
#extractThis  = '1548338989.260526'
extractThis = extractThis.split('/')[-1] if extractThis.startswith('/') else extractThis
loadThis = extractThis if extractThis.startswith('/') else f"{dataPath}{extractThis}"
data   = IO.DataLoader(loadThis)


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
# %%
fig, ax  = subplots(figsize = (10, 10), frameon = False)
ax.set(xticks = [], yticks = [])
positions = nx.nx_agraph.graphviz_layout(model.graph, prog = 'neato', \
                                         )
#                                         root = sorted(dict(model.graph.degree()).items())[0][0])

positions = {node : tuple(i * .01 for i in pos) for node, pos in positions.items() }
p = dict(layout = dict(scale = 1),\
         circle = dict(\
                     radius = 10000),\
         annotate = dict(fontsize = 50000000)
         )
p = {}
plotz.addGraphPretty(model, ax, positions, \
                     **p,\
                     )
#nx.draw(model.graph, positions, ax = ax)
ax.axis('equal')
ax.set(xticks = [], yticks = [])
ax.axis('off')
fig.show()
savefig(figDir + 'network.eps', dpi = 1000)

# %%


# data matrix
#dd       = zeros((NTRIALS, NODES, DELTAS, COND))

# extract data for all nodes
information_impact = '$\mu_i$'
causal_impact      = '$\delta_i$'
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
#sys.exit()
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
                         xlabel = r'Information impact ($\mu_i$)',\
                         ylabel = r'Causal impact ($\delta_i$)'\
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

        tx, ty = aucs[[0, nudge], temp].reshape(COND, -1).copy()
        ddof = tx.size - 2

        ridge.fit(tx[:, None], ty)

        p = 2 * ( 1 - scipy.stats.t.cdf(abs(ridge.coef_), ddof) )

        smax = aucs[[0, nudge], temp].ravel().max(0)
        smin = aucs[[0, nudge], temp].ravel().min(0)

        r = ridge.score(tx[:, None], ty)

        f, p = f_regression(tx[:, None], ty)
        r, p  =scipy.stats.kendalltau(tx, ty)
        rsq = r ** 2
        # plot regression
#        print(r, p )
        if p < pval / pcorr:
#            print(p, )
            tx = linspace(smin, smax)
            ty = ridge.predict(tx[:, None]) * .8
            tax.plot(tx, ty, color = '#a2a2aa', linestyle = '--', alpha = 1)
            tax.text(1, .1, \
             f'$R^2$={rsq:.1e}\np={p:.1e}', \
             fontdict  = dict(fontsize = 10),\
             transform = tax.transAxes,\
             horizontalalignment = 'right', \
             verticalalignment   = 'bottom')
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
from functools import partial
centralities = dict(\
                    deg_w  = partial(nx.degree, weight = 'weight'), \
#                    close  = partial(nx.closeness_centrality, distance = 'weight'),\
                    bet    = partial(nx.betweenness_centrality, weight = 'weight'),\
                    ev     = partial(nx.eigenvector_centrality, weight = 'weight'),\
                    ic     = partial(nx.information_centrality, weight = 'weight'),\
#             cfl = partial(nx.current_flow_betweenness_centrality, weight = 'weight'),
             )


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
    mainax.set_xlabel('Causal impact', labelpad = 50)
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
    fig.canvas.draw()
    #tax.legend(['Information impact', 'Causal impact'], loc = 'upper left', \
    #  bbox_to_anchor = (1.01, 1), borderaxespad = 0)
    subplots_adjust(wspace = .1, hspace = .1)
    #ax.set(xscale = 'log')
    fig.savefig(figDir + f'causal_cent_ii_{conditionLabels[condition]}.eps', \
               format = 'eps', dpi = 1000)



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
labels = 'max $\mu_i$\tdeg\tclose\tbet\tev'.split('\t')

maxT = argmax(targets,  -1)

lll = 'max |$\mu_i$|\t'
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
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import accuracy_score

from sklearn.model_selection import GridSearchCV
import pandas as pd
#clf = linear_model.LogisticRegression(\
#                                        solver      = 'lbfgs',\
#                                        multi_class = 'multinomial',\
#                                        max_iter = 1000, \
#

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
#%% randomforrest
from sklearn import model_selection
groups = arange(N + 1)
cv = model_selection.LeaveOneGroupOut()
cv.get_n_splits(features, yy[..., 0], groups = groups)

ty = yy[..., 0]
clf = RandomForestClassifier(n_estimators = 100, \
                             njobs = -1,\
                             scoring = 'accuracy')

scores = zeros((ty.shape[0]))
for idx, (train, test) in enumerate(cv.split(ty)):
    xi, xj = features[train], features[test]
    yi, yj = ty[test], ty[test]
    clf.fit(xi, yi)
    scores[idx] = clf.score(xj, yj)


print(score.mean())
clf.fit(features, ty)
print(clf.feature_importances_)
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
subplots_adjust(hspace = .5)
allpredset = set(featureClass.as_matrix().flatten())
alltrueset = set(y.as_matrix().flatten())
mapper = {lab : jdx for jdx, lab in enumerate(alltrueset.union(allpredset))}
fig, ax = subplots(1, 1)
#mainax  = fig.add_subplot(111,\
#                              frameon = False,\
#                              xticks  = [],\
#                              yticks  = [],\
#                              ylabel  = f'{fc}\n\n')
for n, fc in enumerate(featureClass):

    nn = len(mapper)
    conf = zeros((nn, nn))
    tax = ax
    for cond, tc in enumerate(y):
        for true, pred in zip(y[tc], featureClass[fc]):
            xi = mapper[true]
            yi = mapper[pred]
            c = 0
            if true and pred:
                c = 1
            conf[xi, yi] += c
    conf = (conf.ravel() / y.shape[0]).reshape(nn,nn)
    conf[isfinite(conf) == False] = 0
    print(conf)

    h = tax.imshow(conf, cmap = cm.plasma,\
                   vmin = 0,\
                   vmax = 1)
    tax.set(\
        yticklabels = mapper, \
        xticklabels = mapper, \
        xticks      = arange(nn),\
        yticks      = arange(nn),\
        )
    tax.set_xlabel('Predicted', labelpad = 5)
    tax.set(title = tc + '\n')



# %%
#
#rcParams['axes.labelpad'] = 40
#fig, ax = subplots(1, 2, sharey = 'all')
#mainax = fig.add_subplot(111,\
#                         xticks = [],\
#                         yticks = [],\
#                         frameon = False)
#mainax.set(ylabel = 'P(C = x | max $\mu_i$)', \
#           xlabel = 'max $\mu_i$')
#
#dmin, dmax = features.min(), features.max()
#dr = linspace(dmin, dmax, 100)
#cmax = scores.max(1).argmax(0)
#for cond in range(COND):
#    tax = ax[cond]
#    clf.set_params(C = cs[cmax[cond]])
#
#    clf.fit(scipy.stats.zscore(features[:, [0]]), tar[condLabels[cond]])
#    pr = clf.predict_proba(dr.reshape(-1, 1))
#    prr= clf.predict_proba(features[:, 0].reshape(-1, 1))
#
#    if cond == 0:
#        print(prr)
#    for kidx, k  in enumerate(clf.classes_):
#        idx = where(tar[condLabels[cond]] == k)[0]
#        print(idx.size, k)
#        tax.scatter(features[idx, 0], prr[idx, kidx], \
#                    color = colors[k], \
#                    label = model.rmapping[k])
#
#        tax.plot(dr, pr[:, kidx], color = colors[k])
#    tax.legend(loc = 'upper right', \
#               title_fontsize = 11, \
#               ncol = 1)
#
#    tax.set(title = condLabels[cond])
#subplots_adjust(wspace = 0)
#fig.savefig(figDir + 'best_estimator.eps')

## %%
#from sklearn import cluster
#from sklearn import mixture
#
#nclus = arange(2, model.nNodes - 1)
#
#tmp = estimates[...,0, :].reshape(-1, 1)
#y   = moveaxis(targets, -2, -1).reshape(-1, COND)
#x   = hstack((tmp, y)).reshape(NTEMPS, NTRIALS * NODES, COND + 1)
#
#from sklearn.metrics import silhouette_samples, silhouette_score
#
#
#gs = dict(\
#          height_ratios = ones(NODES  - nclus.min()-1),\
#          width_ratios  = [2, 1])
#from sklearn.preprocessing import StandardScaler
#
#scores_ = zeros((NTEMPS, nclus.size, COND))
#for temp in range(NTEMPS):
#    for cond in range(COND):
#        fig, ax = subplots(NODES  - nclus.min() - 1, 2, \
#                           sharex = 'col', sharey = 'col', \
#                           gridspec_kw = gs, \
#                           figsize = (10, 30))
#
#        mainax = fig.add_subplot(111, xticks = [],\
#                                 yticks = [],\
#                                 frameon = False)
#        mainax.set_title(f'T={temps[temp]}\n')
#    #    mainax = fig.add_subplot(121, xticks = [],\
#    #                             yticks = [],\
#    #                             frameon = False, \
#    #                             )
#        mainax.set_ylabel('Z-scored $\delta_i$', labelpad = 40)
#
#
#        subplots_adjust(wspace = 0)
#        for idx, n in enumerate(nclus):
#            tax = ax[idx, 1]
#            clf = cluster.KMeans(n)
#    #        clf = mixture.BayesianGaussianMixture(n, covariance_type = 'full')
#            xi = x[temp, :, [0, cond + 1]].T
#            xi = StandardScaler().fit_transform(xi)
#            ypred = clf.fit_predict(xi)
#            sil = silhouette_score(xi, ypred)
#            sils= silhouette_samples(xi, ypred)
#
#            scores_[temp, idx, cond] = sil
#            tax.set_title(f'N={n}')
#            low = 2
#            for ni in range(n):
#                tmp_s = sils[ypred == ni]
#                tmp_s.sort()
#                high = low + tmp_s.size
#
#                color = cm.tab20(ni / n)
#                color = colors[ni]
#                tax.fill_betweenx(arange(low, high), \
#                                  0, tmp_s, color = color,)
#                low = high + 2
#    #        tax.set(yscale = 'log')
#            tax.axvline(sil, color = 'red', linestyle = '--')
#            tax.set(yticks = [])
#            tax = ax[idx, 0]
#            tax.scatter(*xi.T, c = colors[ypred])
#        ax[-1, 0].set_xlabel('Zscored $\mu_i$', labelpad = 15)
#        ax[-1, 1].set_xlabel('Sillouette score', labelpad = 15)
#        fig.savefig(figDir + f'T={round(temps[temp], 2)}_kmeans{cond}.eps')
#
## %%
#gs = dict(\
#          width_ratios = [3, 1, 1])
##fig, ax = subplots(2, NTEMPS, gridspec_kw = gs)
#
#fig = figure()
#width = .3
#s = (2, NTEMPS)
#
#
#for j in range(COND):
#    for i in range(NTEMPS):
#        main = subplot2grid(s, (0,j), colspan = 1)
#
#        tax = subplot2grid(s, (1, i))
#        y = scores_[i, :, j]
#        x = nclus + i * width
#        idx = argmax(y)
#
#        main.bar(x, y,  width = width,\
#               label = round(temps[i], 2), color = colors[i])
#        main.plot(x[idx], y[idx] + .05, '*',color = colors[i],\
#                  markersize = 20)
#
##        clf = cluster.KMeans(nclus[idx])
##        clf.fit(xi, yi)
##[ax.plot(nclus, s, label = round(l,2)) for s, l in zip(scores_, temps)]
#main.legend(title = 'Temperature')
#main.set(xlabel = 'Number of clusters', \
#       ylabel = 'Silloute score', \
#       xticks = nclus + 1/3 * NTEMPS * width,\
#       xticklabels = nclus)
#
## %%DBSCAN
#from sklearn import cluster
#
#
#nclus = arange(2, model.nNodes - 1)
#
#tmp = estimates[...,0, :].reshape(-1, 1)
#y   = moveaxis(targets, -2, -1).reshape(-1, COND)
#x   = hstack((tmp, y)).reshape(NTEMPS, NTRIALS * NODES, COND + 1)
#
#from sklearn.metrics import silhouette_samples, silhouette_score
#
#
#gs = dict(\
#          height_ratios = ones(NTEMPS),\
#          width_ratios  = [2, 1])
#from sklearn.preprocessing import StandardScaler
#
#scores_ = zeros((NTEMPS, nclus.size))
#fig, ax = subplots(NTEMPS, 2, \
#                   sharex = 'col', sharey = 'col', \
#                   gridspec_kw = gs, \
#                   )
#subplots_adjust(wspace = 0)
#for temp in range(NTEMPS):
#
#    mainax = fig.add_subplot(111, xticks = [],\
#                             yticks = [],\
#                             frameon = False)
#    mainax.set_title(f'T={temps[temp]}')
#    tax = ax[temp, 1]
#    clf = cluster.DBSCAN(eps = .35, min_samples = 20, \
#                         metric = 'l1')
#    xi = x[temp, :, [0, 1]].T
#    xi = StandardScaler().fit_transform(xi)
#    ypred = clf.fit_predict(xi)
#    sil = silhouette_score(xi, ypred)
#    sils= silhouette_samples(xi, ypred)
#
#    scores_[temp, idx] = sil
#    tax.set_ylabel(f'N={len(set(clf.labels_))}', rotation = 0)
#    low = 2
#    for ni in set(clf.labels_):
#        tmp_s = sils[ypred == ni]
#        tmp_s.sort()
#        high = low + tmp_s.size
#
##        color = cm.tab20(ni / len(clf.labels_))
#        color = colors[ni]
#        tax.fill_betweenx(arange(low, high), \
#                          0, tmp_s, color = color,)
#        low = high + 2
##        tax.set(yscale = 'log')
#    tax.axvline(sil, color = 'red', linestyle = '--')
#    tax.set(yticks = [])
#    tax = ax[temp, 0]
#    tax.scatter(*xi.T, c = colors[ypred])
#fig, ax = subplots()
#ax.plot(nclus, scores_.T)
## %% visualize data
#
#features  = estimates.argmax(-1)
##features = scipy.stats.zscore(features, axis = 1)
#
#features =  features.reshape(-1, N + 1)
#
#
#clf = RandomForestClassifier(n_estimators = 5, max_features = 1)
#
#
#scores = zeros(NTEMPS * NTRIALS)
#
#importances = zeros((NTEMPS * NTRIALS, N+1))
#rcParams['axes.labelpad'] = 10
#
#titles = 'Underwhelming Overwhelming'.split()
#gs = dict(\
#          height_ratios = [2, 1],\
##          width_ratios  = [.1, .1],\
#          )
#fig, ax = subplots(COND, COND, sharey = 'row', sharex = 'row', \
#                   gridspec_kw = gs)
#
#subplots_adjust(wspace = 0, hspace = .5)
#
#
#for cond in range(COND):
#    y  = targets.argmax(-1)[..., cond].ravel()
#    tax = ax[0, cond]
#    for idx, (train, test) in enumerate(LeaveOneOut().split(features, y)):
#        xi = features[train]
#        yi = y[train]
#        clf.fit(xi, yi)
#        scores[idx] = clf.score(features[test], y[test])
#
#        importances[idx] = clf.feature_importances_
#    x = arange(NTEMPS)
#    s = scores.reshape(NTEMPS, NTRIALS) * 100
#    tax.errorbar(x, s.mean(1), yerr = s.std(axis = 1) , fmt = 'none', ecolor = 'k')
#    tax.bar(x, s.mean(1))
#    tax.set(\
#       xticks = x,\
#       xticklabels = [f'T={round(i,2)}' for i in temps])
#    if cond == 0:
#        tax.set_ylabel('Prediction accuracy\n(%)')
#    tax.tick_params(axis = 'x', rotation = 45)
#    tax.set_title(titles[cond])
#
#    tax = ax[1, cond]
#    x = arange(1, N + 2, )
#    tax.errorbar(x, importances.mean(0), importances.std(0), fmt = 'none')
#    tax.bar(x, importances.mean(0))
#    tax.set(xticks = x, \
#            xticklabels = lll, \
#            )
#    if cond == 0 :
#        tax.set_ylabel('Feature\nimportance')
#    tax.tick_params(axis = 'x', rotation = 45)
#    tax.set_ylim(0, 1.2)
#
#fig.savefig(figDir + 'randomforrest.eps')
##z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
##ax.contourf(xx, yy, z.reshape(xx.shape), alpha = .2)
#
##clf.score(features, targets)
##%% appendix box rejection
#fig, ax = subplots(1, COND, sharey = 'all')
#mainax = fig.add_subplot(111, frameon = False,\
#                         xticks = [],\
#                         yticks = [],\
#                         ylabel = 'Rejection rate(%)'\
#                         )
#rcParams['axes.labelpad'] = 50
#subplots_adjust(wspace = 0)
#x     = arange(NTEMPS) * 3
#
#width = .2
#labels = 'Underwhelming Overwhelming'.split()
#conditions = [f'T={round(i,2)}' for i in temps]
#for cond in range(COND):
#    tax = ax[cond]
#    tax.set_title(labels[cond])
#    tax.set(xticks = x + .5 * width * NODES, \
#            xticklabels = conditions,\
#            )
#    tax.tick_params(axis = 'x', rotation = 45)
#
#    for node in range(NODES):
#        y  = rejections[cond, :, node] * 100
#        tx = x + width * node
#        tax.bar(tx, y, width = width,\
#                color = colors[node], label = model.rmapping[node])
#tax.legend(loc = 'upper left',\
#          bbox_to_anchor = (1, 1),\
#          borderaxespad = 0, \
#          frameon = False)
#fig.savefig(figDir + 'rejection_rate.eps', format = 'eps', dpi = 1000)
## %% appendix plot: fit error comparison
#fig, ax = subplots()
#errors = errors.reshape(-1, NODES)
#subplots_adjust(hspace = 0)
#width= .07
#locs = arange(0, len(errors))
#
#
#
#for node in range(NODES):
#    ax.bar(locs + width * node, \
#           errors[:, node], color = colors[node],\
#           width = width, label = model.rmapping[node])
#
#conditions = [f'T={round(x, 2)}\n{y}' for x in temps for y in nudges]
## set labels per group and center it
#ax.set(yscale = 'log', xticklabels = conditions, \
#       xticks = locs + .5 * NODES * width, ylabel = 'Mean square error')
#ax.tick_params(axis = 'x', rotation = 45)
#ax.legend(loc = 'upper left',\
#          bbox_to_anchor = (1, 1),\
#          borderaxespad = 0, \
#          frameon = False)
#rcParams['axes.labelpad'] = 1
#fig.savefig(figDir + 'fiterror.eps', format = 'eps', dpi = 1000)
