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

#close('all')
style.use('seaborn-poster')
dataPath = f"{os.getcwd()}/Data/"
#dataPath = '/mnt/'
extractThis      = IO.newest(dataPath)[-1]
extractThis      = '1548025318.5751357'
#extractThis      = '1547303564.8185222'
#extractThis  = '1548338989.260526'
extractThis = extractThis if extractThis.startswith('/') else f"{dataPath}{extractThis}"
data   = IO.DataLoader(extractThis)


settings = IO.readSettings(extractThis)
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
DELTAS, EXTRA = divmod(deltas, 2) #use half, also correct for border artefact
COND     = NNUDGE - 1

pulseSizes = list(data[f't={temp}'].keys())

print(f'Listing temps: {temps}')
print(f'Listing nudges: {pulseSizes}')

figDir = '../thesis/figures/'
# %% Extract data

# %% # show mag vs temperature
tmp = IO.loadPickle(f'{extractThis}/mags.pickle')
fig, ax = subplots()
ax.scatter(tmp['temps'], tmp['mag'], alpha = .2)
ax.scatter(tmp['temperatures'], tmp['magRange'] * tmp['mag'].max(), \
           color = 'red', zorder = 2)

func = lambda x, a, b, c, d :  a / (1 + exp(b * (x - c))) + d # tanh(-a * x)* b + c
a, b = scipy.optimize.curve_fit(func, tmp['temps'], tmp['mag'].squeeze(), maxfev = 10000)
x = linspace(min(tmp['temps']), max(tmp['temps']), 1000)
ax.plot(x, func(x, *a), '--k')
ax.set(xlabel = 'Temperature (T)', ylabel = '|<M>|')
rcParams['axes.labelpad'] = 10
fig.savefig(figDir + 'temp_mag.eps', format = 'eps', dpi = 1000)
# %%
#fig, ax  = subplots(figsize = (400, 200), frameon = False)
#ax.set(xticks = [], yticks = [])
#positions = nx.nx_agraph.graphviz_layout(model.graph, prog = 'neato', \
#                                         )
##                                         root = sorted(dict(model.graph.degree()).items())[0][0])
#
#positions = {node : tuple(i * 1 for i in pos) for node, pos in positions.items() }
#p = dict(layout = dict(scale = 1),\
#         circle = dict(\
#                     radius = 15),\
#         annotate = dict(fontsize = 14)
#         )
#p = {}
##plotz.addGraphPretty(model, ax, positions, \
#                     **p,\
#                     )
#nx.draw(model.graph, positions, ax = ax)
#ax.axis('equal')
#ax.set(xticks = [], yticks = [])
#ax.axis('off')
#fig.show()
#savefig(figDir + 'psychonetwork.eps', dpi = 1000)

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
    if '{}' in fileName:
        # load control; bias correct mi
        control = IO.loadData(fileName)
        graph   = control.graph
        # panzeri-treves correction
        cpx = control.conditional
        N   = repeats
        mi  = control.mi
        rs  = zeros(mi.shape)
        for key, value in cpx.items():
#                xx  = value[0] if isinstance(value, list) else value
            for zdx, deltaInfo in enumerate(value):
                for jdx, nodeInfo in enumerate(deltaInfo):
                    rs[zdx, jdx] += plotz.pt_bayescount(nodeInfo, repeats) - 1
#                rs += array([[plotz.pt_bayescount(k, repeats) - 1 for k in j]\
#                              for j in value])
        Rs = array([[plotz.pt_bayescount(j, repeats) - 1 for j in i]\
                     for i in control.px])

        bias = (rs - Rs) / (2 * repeats * log(2))
        corrected = mi - bias
        data[0, trial, temp]  = corrected[:DELTAS - EXTRA, :].T
    else:
        # load control
        targetName = fileName.split('_')[-1] # extract relevant part
        jdx = [model.mapping[int(j)] if j.isdigit() else model.mapping[j]\
                             for key in model.mapping\
                             for j in re.findall(str(key), re.sub(':(.*?)\}', '', targetName))]
        jdx = jdx[0]
        useThis = fidx - node
        control = IO.loadData(fileNames[useThis])
        # load nudge
        sample  = IO.loadData(fileName)
        px      = sample.px
#        impact  = stats.hellingerDistance(control.px, px)
        impact  = stats.KL(control.px, px)
        # don't use +1 as the nudge has no effect at zero
        redIm = nanmean(impact[DELTAS + EXTRA:], axis = -1).T
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
    print(a, b, c, COND * NODES + 1 )
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

#syF         = symbols[1] * sy.exp(- symbols[2] * symbols[0]) +\
# symbols[3] * sy.exp(- symbols[4] * (symbols[0] - symbols[5]))
#print(syF)
##syF         = symbols[1] * sy.exp(- symbols[2] * symbols[0])
#func        = lambdify(symbols, syF, 'numpy')
func        = double
p0          = ones((func.__code__.co_argcount - 1)); # p0[0] = 0
fitParam    = dict(maxfev = int(1e6), \
                   bounds = (0, inf), p0 = p0,\
                   jac = 'cs')

settings = IO.readSettings(extractThis)
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
ax[1, 2].set_ylabel("$D_{KL}(P \vert\vert P')", labelpad = 5)
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
from mpl_toolkits.axes_grid1 import make_axes_locatable
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
def worker(sample):
    auc = zeros((len(sample), 2))
    coeffs, errors = plotz.fit(sample, func, params = fitParam)
    for nodei, c in enumerate(coeffs):
    #        tmp = syF.subs([(s, v) for s,v in zip(symbols[1:], c)])
    #        tmp = sy.integrals.integrate(tmp, (abc.x, 0, DELTAS))
        F   = lambda x: func(x, *c)
        tmp, _ = scipy.integrate.quad(F, 0, inf)
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
fig, ax = subplots(3, 2)
subplots_adjust(hspace = 0, wspace = .2)
mainax = fig.add_subplot(111, frameon = False, \
                         xticks = [], yticks = [],\
                         xlabel = r'Information impact ($\mu_i$)',\
                         ylabel = r'Causal impact ($\delta_i$)'\
                         )
out = lof(n_neighbors = NTRIALS)

aucs = aucs_raw.copy()
thresh = 2.5
clf = MinCovDet()

labels = 'Underwhelming\tOverwhelming'.split('\t')
rejections = zeros((COND, NTEMPS, NODES))
raw = False
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
                if raw:
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
                    if raw:
                        tax.scatter(*aucs_raw[[0, nudge], temp, outlier, idx], \
                                color = colors[idx], \
                                alpha = 1, marker = 's')
            except:
                continue

#            tax.scatter(*tmp.mean(1), color = 'k', s = 50, zorder = 5)
            alpha = .1 if raw else 1
            tax.scatter(*aucs[[0, nudge], temp, :, idx], \
                        color = colors[idx], \
                        label = node, alpha = alpha, \
                        linewidth = 1)
#            tax.scatter(*tmp[:, ident < 0], s = 100, marker = 's')
        ticks = tax.get_yticks()
#        tax.set_yticks([ticks[0], ticks[-1]], False)
#        tax.set_yticklabels([smin.round(), smax.round()])
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

out = 'causal_ii_scatter_raw' if raw else 'causal_ii_scatter_corrected'
out += '.png' if raw else '.eps'
savefig(figDir + out, dpi = 1000)

# %% information impact vs centrality
from functools import partial
centralities = dict(deg      = nx.degree, \
             close   = nx.closeness_centrality,\
             bet = partial(nx.betweenness_centrality, weight = 'weight'),\
             ev = partial(nx.eigenvector_centrality, weight = 'weight'),\
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
    fig, ax = subplots(4, 3, sharex = 'all', sharey = 'row')
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
        for ni, centF in enumerate(centralities.values()):
            tmp = dict(centF(model.graph))
            centLabel, centVal= sorted(tmp.items(), key = lambda x: abs(x[1]))[-1]
            centEstimate = model.mapping[centLabel]
            print(ni + 1, centLabel, centEstimate)
            percentages[temp, ni + 1, cond] =  equal(\
                       centEstimate * ones(NTRIALS), \
                       drivers[cond + 1, temp]).mean()

            maxestimates[temp, :, ni + 1] = model.mapping[centLabel] * ones(NTRIALS)
            for k, v in tmp.items():
                estimates[temp, :, ni + 1, model.mapping[k]] = v * ones((NTRIALS))
print(percentages)
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
                         ylabel = 'Predicition accuracy')
#mainax.set(ylabel = "Prediction accuracy")
subplots_adjust(wspace = 0)
width = .4
xloc  = arange(NTEMPS) * 2
conditions = [f'T={round(x, 2)}\n{y}' for x in temps for y in nudges]
condLabels = 'Underwhelming Overwhelming'.split()
labels = 'max $\mu_i$\tdeg\tclose\tbet\tev'.split('\t')

maxT = argmax(targets,  -1)

lll = 'Information impact\t'
ii = '\t'.join(i for i in centralities)
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

# %% cross validation
from sklearn.feature_selection import SelectKBest, RFE, RFECV
from sklearn.svm import SVC
from sklearn.feature_selection import chi2, mutual_info_classif
from sklearn.model_selection import StratifiedKFold, LeaveOneOut


from sklearn.ensemble import RandomForestClassifier
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import accuracy_score
features  = estimates.max(-1).reshape(-1, N + 1)
#features  = maxestimates.reshape(-1, N + 1) + 1
features  = repeat(features, COND, 0)

featTarget = maxT.reshape(-1) + 1 

svc = SVC(kernel = 'linear')

#featTarget= targets.argmax(-1).reshape(-1, COND)

clf= SelectKBest(chi2, k = 1).fit(\
                features, featTarget)

rfe = RFE(estimator = svc, n_features_to_select=1 ,step = 1)
rfe.fit(features, featTarget)

ranking = rfe.ranking_

rfecv = RFECV(estimator = svc, step = 2, cv = LeaveOneOut(),\
              scoring = 'accuracy')
rfecv.fit(features, featTarget)
print(rfecv.n_features_)

fig, ax = subplots()
x = arange(1, rfecv.grid_scores_.size + 1)
ax.plot(x, rfecv.grid_scores_)
print(clf.scores_)

# %%
clf = RandomForestClassifier(n_estimators=10000, \
                             random_state=0, n_jobs=-1)
sfm = SelectFromModel(clf, threshold=0.15)
xtrain, xtest, ytrain, ytest = train_test_split(features, featTarget, test_size = .1, random_state = 0)

sfm.fit(xtrain, ytrain)
for i in sfm.get_support(indices = True):
    print(i)
clf.fit(xtrain, ytrain)
pred = clf.predict(xtest)
print(accuracy_score(ytest, pred))

xtrain_ = sfm.transform(xtrain)
xtest_  = sfm.transform(xtest)

clf = RandomForestClassifier(n_estimators=10000, \
                             random_state=0, n_jobs=-1)

clf.fit(xtrain_, ytrain)

pred = clf.predict(xtest_)
print(accuracy_score(ytest, pred))
 #%%
#from sklearn.model_selection import LeaveOneOut, StratifiedKFold
#from sklearn import svm, metrics, linear_model as lm
#
##clf = svm.SVC(decision_function_shape = 'ovr', gamma = 'scale')
##clf = lm.LogisticRegression(\
##                            solver = 'lbfgs',\
##                            multi_class = 'multinomial')
#clf = lm.LogisticRegression(C = 1, solver = 'lbfgs')
#from sklearn.model_selection import cross_val_score
#
#y = zeros(NTEMPS * NTRIALS)
##idx = np.argmax(estimates, -1).reshape(-1, N+1)[...,0] == np.argmax(targets, -1).reshape(-1, COND)[..., 0]
#y = targets.argmax()
##y[idx] = 1
#y = y.reshape(-1)
#ss = zeros((N+1, NTEMPS, COND))
#from sklearn.model_selection import ShuffleSplit
#
#cv = ShuffleSplit(n_splits = 3, test_size = 0.2, random_state = 0)
#cv =  LeaveOneOut()
#
#n_splits = 14
#cv = StratifiedKFold(n_splits = n_splits, shuffle = True)
#fig, ax = subplots(3, 2)
#for cond in range(COND):
#    for i in range(N + 1):
#        for temp in range(NTEMPS):
#            xi = zeros(NTRIALS)
#            for tr in range(NTRIALS):
#                m      = maxestimates[temp, tr, i]
#                xi[tr] = estimates[temp, tr, i, m]
##            xi = estimates[temp, :, [i], m].T
#            yi = equal(maxT[temp, :, cond], maxestimates[temp, :, i])
##            print(cond, i, yi.mean())
#    #        ss[i] = cross_val_score(clf, xi, yi, cv = cv).mean()
#            for trainidx, testidx in cv.split(xi, yi):
#                try:
#    #                print(i, temp, yi[trainidx].mean())
#                    clf.fit(xi[trainidx].reshape(-1, 1), yi[trainidx])
#                    pred =  clf.predict(xi[testidx].reshape(-1, 1))
#
#                    s = 0
#                    xxx = linspace(-100, 100)
#                    yyy = clf.predict_proba(xxx[:, None])
#                    tax = ax[temp, cond]
#                    tax.plot(xxx, yyy[:, 1], color = colors[i])
##
#                    if pred == yi[testidx]:
#                        s = 1
#                    ss[i, temp, cond] += s
#
#                except Exception as e:
#                    ss[i, temp, cond] += yi.mean()
#    #                print(i, temp, yi.mean(), e)
#    #                ss[i, temp, testidx] = 0
#    #
#    #                ss[i, temp, testidx] = yi[trainidx].mean()
#
#
#ss = ss / n_splits
#fig, ax = subplots(COND)
#for ci in range(COND):
#    tax = ax[ci]
#    tax.imshow(ss[..., ci])

# %%
c    = percentages.reshape(-1, COND) * NTRIALS
ddof = (NTEMPS * N - 1) * (COND - 1)

e = (c.sum(1)  * c.sum(0)[:, None]).T / (c.sum())
chi =  nansum((c - e)**2 / e)
#%% appendix box rejection
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
        y  = rejections[:, cond, node] * 100
        tx = x + width * node
        tax.bar(tx, y, width = width,\
                color = colors[node], label = model.rmapping[node])
tax.legend(loc = 'upper left',\
          bbox_to_anchor = (1, 1),\
          borderaxespad = 0, \
          frameon = False)
fig.savefig(figDir + 'rejection_rate.eps', format = 'eps', dpi = 1000)
# %% appendix plot: fit error comparison
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


# %%
#percentages[percentages == 0] = 0
#contin = scipy.stats.chi2_contingency
#counts = percentages * NTRIALS
#res = contin(counts, correction = True)
#print(res)


## %%
#conds = f'II UN OV'.split()
#for i in centralities:
#    conds.append(f'{i}')
#gLabels = []
#for temp in temps:
#    for cond in conds:
#        gLabels.append(f'{round(temp,2)}_{cond}')
#print(gLabels)


# %% statistical tests
#
#methods =  'II DEG CLOS BET EV'.split()
#cis     = 'UN OV'.split()
#levels  = [[], [], []]
#
#import pandas as pd
#for temp in temps:
#    for method in methods:
#        for ci in cis:
#            levels[0].append(f'T={round(temp,2)}')
#            levels[2].append(method)
#            levels[1].append(ci)
#names = 'Temperature Method CI'.split()
#index = pd.MultiIndex.from_tuples(zip(*levels), names = names)
#s = pd.Series(percentages.ravel(), index = index)
#print(s)
# %%
#driver = pd.DataFrame(all_drivers.T, columns = gLabels)
#print(driver)
## kruskal wallis
#results = vstack((percentage, centApprox.reshape(-1, COND)))
#
#H, p  = scipy.stats.kruskal(*results.T)
#print(f"H = {H}, p = {p:e}")
#import scikit_posthocs as sp, pandas as pd
#indx = array([1, 8, 9, 10, 11])
#
##df = pd.DataFrame(tmp, columns = "II Deg Close Betw Ev".split())
##df = df.melt(var_name = 'groups', value_name = 'values')
## %% post hoc
#df = driver.melt(var_name = 'groups')
#print(df)
#pc = sp.posthoc_conover(df, group_col = 'groups', val_col = 'value', p_adjust = 'bonferroni')
#
#heatmap_args = {'linewidths': 0.25, 'linecolor': '0.5', 'clip_on': False, 'square': True, 'cbar_ax_bbox': [0.80, 0.35, 0.04, 0.3]}
#fig, ax = subplots()
#sp.sign_plot(pc , **heatmap_args)
# %%
#import scikit_posthocs as sp
#percentage = percentage = array([i == target for i in rankings.T]).mean(1)
#
#test = hstack((rankings, target[:, None]))
##test2 =
#res =  scipy.stats.kruskal(*test.T)
#print(res)

 # %% compute concistency
#
#bins = arange(-.5, model.nNodes + .5)
#cons = lambda x : nanmax(histogram(x, bins , density = True)[0])
#
#ranking, maxim = stats.rankData(aucs)
#consistency = array([ [cons(i) for i in j] for j in maxim.T])
## plot consistency of the estimator
#naive = (maxim[...,0] == maxim[...,1])
