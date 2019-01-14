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
import os, re, networkx as nx, scipy

#close('all')
style.use('seaborn-poster')
dataPath = f"{os.getcwd()}/Data/"
#dataPath = '/mnt/'
extractThis      = IO.newest(dataPath)[-1]
#extractThis      = '1539187363.9923286'    # th.is is 100
#extractThis      = '1540135977.9857328'
extractThis = extractThis if extractThis.startswith('/') else f"{dataPath}{extractThis}"
data   = IO.DataLoader(extractThis)

# thetas = [10**-i for i in range(1, 20)]
thetas  = logspace(log10(.9), log10(finfo(float).eps), 100)
#thetas  = array([.5, .1, .01, .001])


temps    = list(data.keys())
temp     = temps[0]
pulseSize= list(data[temp].keys())[1]

print(f'Listing temps: {temps}')
print(f'Listing nudges: {pulseSize}')


# %% Extract data

# Draw graph ; assumes the simulation is over 1 type of graph
from Models.fastIsing import Ising
control  = IO.loadData(data[temp]['control'][0]) # TEMP WORKAROUND
model    = Ising(control.graph)
fig, ax  = subplots()
plotz.addGraphPretty(model, ax = ax)
ax.axis('off')
fig.show()

settings = IO.readSettings(extractThis)
deltas   = settings['deltas']
repeats  = settings['repeat']


NTRIALS  = len(data[temp]['control'])
NODES    = model.nNodes
DELTAS   = deltas // 2 - 1 # throw away half for nudge
COND     = 2
THETAS   = thetas.size

# data matrix
dd       = zeros((NTRIALS, NODES, DELTAS, COND))

# extract data for all nodes
for condition, fileNames in data[temp][pulseSize].items():
    for idx, fileName in enumerate(fileNames):
        control = IO.loadData(data[temp]['control'][idx])
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
        dd[idx, ..., 0] = corrected[:DELTAS, :].T
        
        sample  = IO.loadData(fileName)
        px      = sample.px
#        impact  = stats.hellingerDistance(control.px, px)
        impact  = stats.KL(control.px, px)
#            impact = nanmean(tmp, axis = -1)
#            print(impact)
        # don't use +1 as the nudge has no effect at zero
        redIm = nanmean(impact[DELTAS + 2:], axis = -1).T
#            print(impact)
        # TODO: check if this works with tuples (not sure)
        jdx = [model.mapping[int(j)] if j.isdigit() else model.mapping[j]\
            for key in model.mapping\
            for j in re.findall(str(key), re.sub(':(.*?)\}', '', condition))]
#            print(model.rmapping[jdx[0]], tmp[deltas//2, jdx])
        dd[idx, jdx, ...,  1] = redIm.squeeze().T
        


# define color swaps
colors = cm.tab20(arange(NODES))
# swap default colors to one with more range
rcParams['axes.prop_cycle'] = cycler('color', colors)
# %% extract root from samples

# fit functions
double = lambda x, a, b, c, d, e, f: a + b * exp(-c*(x)) + d * exp(- e * (x-f))
double_= lambda x, b, c, d, e, f: b * exp(-c*(x)) + d * exp(- e * (x-f))
single = lambda x, a, b, c : a + b * exp(-c * x)
single_= lambda x, b, c : b * exp(-c * x)
special= lambda x, a, b, c, d: a  + b * exp(- (x)**c - d)
func        = double
p0          = ones((func.__code__.co_argcount - 1)); # p0[0] = 0
fitParam    = dict(maxfev = int(1e6), bounds = (0, inf), p0 = p0)

settings = IO.readSettings(extractThis)
repeats  = settings['repeat']


# %% normalize data
from scipy import ndimage
zd = dd;
#zd = ndimage.filters.gaussian_filter1d(zd, 2, axis = -2)
#zd = ndimage.filters.gaussian_filter1d(zd, .2, axis = 0)


# scale data 0-1 along each sample (nodes x delta)
rescale = True
#rescale = False
if rescale:
    zd = zd.reshape(zd.shape[0], -1, zd.shape[-1])
    MIN, MAX = zd.min(axis = 1), zd.max(axis = 1)
    MIN = MIN[:, newaxis, :]
    MAX = MAX[:, newaxis, :]
    zd = (zd - MIN) / (MAX - MIN)
    zd = zd.reshape(dd.shape)
    
thresh = 1e-4
#zd[zd <= thresh] = 0
zd[zd <= finfo(zd.dtype).eps] = 0 # remove everything below machine error
# show means with spread
fig, ax = subplots(1, 2)
mainax  = fig.add_subplot(111, frameon = 0)
mainax.set(xticks = [], yticks = [])
mainax.set_title(f'{temp}\n\n')
mainax.set_xlabel('Time[step]', labelpad = 40)
x  = arange(DELTAS)
xx = linspace(0, 1 * DELTAS, 1000)
sidx = 1.96
labels = 'MI IMPACT'.split()

from matplotlib.ticker import FormatStrFormatter
mins, maxs = zd.reshape(-1, COND).min(0), zd.reshape(-1, COND).max(0)

mins_, maxs_ = zd.min(0), zd.max(0)
means, stds  = zd.mean(0), zd.std(0, ddof = 1)

for cidx in range(COND):
    # compute mean fit
    meanCoeffs, meanErrors = plotz.fit(means[..., cidx].T, func, params = fitParam)
    for node, idx in sorted(model.mapping.items(), key = lambda x : x[1]):
        # plot the raw data
        ax[cidx].errorbar(x, means[idx, :, cidx], fmt = '.',\
          yerr = sidx * stds[idx, :, cidx], markersize = 20, \
          label = node,\
          color = colors[idx])
        # plot mean fit
        ax[cidx].plot(xx, func(xx, *meanCoeffs[idx]),\
          color = colors[idx], alpha = .5, \
          markeredgecolor = 'black')
#        ax[cidx].set(yscale = 'log', xscale = 'log')

        # fill the standard deviation from mean
#        ax[cidx].fill_between(x, a, b, color = colors[idx], alpha  = .4)
#        ax[cidx].yaxis.set_major_formatter(FormatStrFormatter('%1.0e'))

#    ax[cidx].set(yticks = (0, maxs_[..., cidx].max()))
#    ax[cidx].ticklabel_format(axis = 'y', style = 'sci', scilimits = (0, 2))
    ax[cidx].set_title(labels[cidx])
#    ax[cidx].set(xscale = 'log', yscale = 'log'
from mpl_toolkits.axes_grid1 import make_axes_locatable
ax[-1].legend(\
  title = 'Node', title_fontsize = 20, loc = 'upper left', \
  bbox_to_anchor = (1.01, 1), borderaxespad = 0)
fig.show()
# %% estimate impact
aucs       = zeros((NTRIALS, COND, NODES))

from scipy import interpolate
from time import sleep

COEFFS = zeros((COND, NTRIALS, NODES, p0.size))
x = arange(DELTAS)

#lim = inf
lim = DELTAS
#lim = np.inf
print('Estimating area under the curve')
for samplei, sample in enumerate(zd):
    for condi, s in enumerate(sample.T):
        coeffs, errors = plotz.fit(s, func, params = fitParam)
        COEFFS[condi, samplei, ...] = coeffs
        for nodei, c in enumerate(coeffs):
            output =  integrate.quad(func, 0, \
                                           lim, args = tuple(c), \
                                           full_output = 1)
            if len(output) == 3:
                auc, err, output = output
            elif len(output) == 4:
                auc, err, output, message = output
                print(f'{nodei} {err} {auc}')
            aucs[samplei, condi, nodei] = auc
# %% show idt auc vs impact auc
fig, ax = subplots()
for idx, node in sorted(model.rmapping.items(), key = lambda x : x[0]):
    c = colors[idx]
    i = aucs[..., idx].T
    ax.scatter(*i, color = c, label = model.rmapping[idx])
    ax.scatter(*median(i, 1), marker = 's', color = c, edgecolors = 'k')
    ax.scatter(*mean(i, 1), marker = '^', color = c, edgecolors = 'k')
ax.legend(title = 'Node', loc = 'upper right', bbox_to_anchor = (1.15, 1))
#ax.set(yscale = 'symlog')

 # %% compute concistency

bins = arange(-.5, model.nNodes + .5)
cons = lambda x : nanmax(histogram(x, bins , density = True)[0])

ranking, maxim = stats.rankData(aucs)
consistency = array([ [cons(i) for i in j] for j in maxim.T])
# plot consistency of the estimator
naive = (maxim[...,0] == maxim[...,1])

# %% frequency heatmaps consistency
tmp = array([ \
     [\
      histogram(j, bins = bins, density = True)[0]\
      for j in i.T ]
     for i in maxim.T])
fig, ax = subplots(1, 2, sharey = 'all')
mainax  = fig.add_subplot(111, frameon = False,
                          xticks = [], yticks = [])
for axi, t in zip(ax, tmp):
    h = axi.imshow(t.T, aspect = 'auto', vmin = 0, vmax = 1)

colorbar(h, ax = axi, label = 'frequency')
mainax.set_xlabel(r'$\theta$', labelpad = 30)
# %%
def colorbar(mappable, **kwargs):
    ax = mappable.axes
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    return fig.colorbar(mappable, cax=cax, **kwargs)
fig, ax = subplots(1, COND, sharex = 'all', sharey = 'all')

mainax = fig.add_subplot(111, frameon = False)
mainax.set_title('Consistency of ranking over trials')
mainax.set(**dict(\
                  xticks = [],\
                  yticks = []))
mainax.set_xlabel('Trial number', labelpad = 10)
mainax.set_ylabel('Node idx', labelpad = 50)
subplots_adjust(wspace = .2)
for cidx in range(COND):
    h = ax[cidx].imshow(ranking[:, cidx, :].T)
    cbar = colorbar(h, ax = ax[cidx], anchor = (.5, 0))
cbar.set_label('Ranking')
# %%


from functools import partial
centralities = dict(degree      = nx.degree_centrality, \
             eigenvector = partial(nx.eigenvector_centrality, weight = 'weight'),\
             closeness   = nx.closeness_centrality,\
             betweenness = partial(nx.betweenness_centrality, weight = 'weight'),\
             )

fig, ax = subplots(2,2, sharex = 'all')
mainax = fig.add_subplot(111, frameon = False, xticks = [], yticks = [])
mainax.set_xlabel('Information impact', labelpad = 50)
mainax.set_ylabel('Centrality', labelpad = 50)
infImpact = aucs.mean(0)


# get the test statistic
ranking.sort()
target   = ranking[:, 1, -1]
rankings = ranking[:, 0, [-1]]

for i, (cent, cent_func) in enumerate(centralities .items()):
    print(idx)
    centrality = array(list(cent_func(model.graph).values()))
    rankings = hstack((rankings, argsort(centrality)[[-1]] * ones((len(ranking), 1))))
    tax = ax.ravel()[i]
    for idx, (impact, c) in enumerate(zip(infImpact, ['r', 'b'])):
        tax.scatter(impact, centrality)
#        tax.set(xscale = 'log')
    tax.set(title = cent)

tax.legend(['Information impact', 'Causal impact'], loc = 'upper left', \
  bbox_to_anchor = (1.01, 1), borderaxespad = 0)
subplots_adjust(hspace = .8)
#ax.set(xscale = 'log')

for label in get_figlabels():
    savefig(f'Figures/{label}')
show()

# %%
percentage = percentage = array([i == target for i in rankings.T]).mean(1)

test = hstack((rankings, target[:, None]))
#test2 =
res =  scipy.stats.kruskal(*test.T)
print(res)
