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

import IO, os, plotting as plotz, stats, re
close('all')
style.use('seaborn-poster')
dataPath = f"{os.getcwd()}/Data/"
extractThis      = IO.newest(dataPath)[-1]
#extractThis      = '1539187363.9923286'    # th.is is 100
#extractThis      = '1540135977.9857328'
extractThis = extractThis if extractThis.startswith('/') else f"{dataPath}{extractThis}" 

data   = IO.extractData(extractThis)

# thetas = [10**-i for i in range(1, 20)]
thetas  = logspace(log10(.9), log10(finfo(float).eps), 100)
#thetas  = array([.5, .1, .01, .001])
temp    = list(data.keys())[0]

from fastIsing import Ising
model   = Ising(data[temp]['{}'][0].graph)

# %% Extract data
settings = IO.readSettings(extractThis)
deltas   = settings['deltas']
repeats  = settings['repeat']
controls = array([i.mi for i in data[temp]['{}']])
roots    = zeros((len(controls), model.nNodes, len(thetas), 2))

colors = cm.tab20(arange(model.nNodes))
# swap default colors to one with more range
rcParams['axes.prop_cycle'] = cycler('color', colors)

NSAMPLES, NODES, DELTAS, COND = len(controls), model.nNodes, deltas // 2, 2
THETAS = thetas.size
dd = zeros((NSAMPLES, NODES, DELTAS, COND), dtype = float32)
for condition, samples in data[temp].items():
    for idx, sample in enumerate(samples):
        if condition == '{}':
            # panzeri-treves correction
            cpx = sample.conditional
            N = repeats 
            
            mi = sample.mi
            rs  = zeros(mi.shape)
            for key, value in cpx.items():
#                xx  = value[0] if isinstance(value, list) else value
                for zdx, deltaInfo in enumerate(value):
                    for jdx, nodeInfo in enumerate(deltaInfo):
                        rs[zdx, jdx] += plotz.pt_bayescount(nodeInfo, repeats) - 1
#                rs += array([[plotz.pt_bayescount(k, repeats) - 1 for k in j]\
#                              for j in value])
            Rs = array([[plotz.pt_bayescount(j, repeats) - 1 for j in i]\
                         for i in sample.px])
            
            bias = (rs - Rs) / (2 * repeats * log(2)) 
            corrected = mi - bias
            dd[idx, ..., 0] = corrected[:deltas // 2, :].T
        else:
            control = data[temp]['{}'][idx].px
            px      = sample.px
#            impact = stats.hellingerDistance(px, control).mean(-1)
            impact = nanmean(stats.KL(control, sample.px), axis = -1)
#            print(impact)
            redIm = impact[deltas // 2  + 1 : ][None, :].T
#            print(impact)
            # TODO: check if this works with tuples (not sure)
            jdx = [model.mapping[int(j)] if j.isdigit() else model.mapping[j]\
                for key in model.mapping\
                for j in re.findall(str(key), re.sub(':(.*?)\}', '', condition))]
            dd[idx, jdx, ...,  1] = redIm.squeeze().T
#dd [dd < finfo(float).eps ] = 0
# %% extract root from samples
            
# fit functions
double = lambda x, a, b, c, d, e, f: a + b * exp(-c*x) + d * exp(- e * (x-f))   
single = lambda x, a, b, c : a + b * exp(-c * x)
single_= lambda x, b, c : b * exp(-c * x)
func   = double
p0          = ones((func.__code__.co_argcount - 1));# p0[0] = 0
fitParam    = dict(maxfev = int(1e4), bounds = (0, inf), p0 = p0)

settings = IO.readSettings(extractThis)
repeats  = settings['repeat']


# %% normalize data
from scipy import ndimage
zd = dd;
#zd = ndimage.filters.gaussian_filter1d(zd, .5, axis = -2)
#zd = ndimage.filters.gaussian_filter1d(zd, 2, axis = 0)
zd[zd < finfo(float).eps] = 0

# scale data 0-1 along each sample (nodes x delta)
#zd = zd.reshape(zd.shape[0], -1, zd.shape[-1])
#
#MIN, MAX = zd.min(axis = 1), zd.max(axis = 1)
#MIN = MIN[:, newaxis, :]
#MAX = MAX[:, newaxis, :]
#zd = (zd - MIN) / (MAX - MIN)
#zd = zd.reshape(dd.shape)

# show means with spread
fig, ax = subplots(1, 2)
mainax  = fig.add_subplot(111, frameon = 0)
mainax.set(xticks = [], yticks = [])
mainax.set_title(temp + '\n\n')
x = arange(deltas // 2)
sidx = 1.
labels = 'MI IMPACT'.split()

from matplotlib.ticker import FormatStrFormatter
mins, maxs = zd.reshape(-1, COND).min(0), zd.reshape(-1, COND).max(0)
for tmp in zip(ax, zd.mean(0).T, zd.std(0).T, labels, mins, maxs):
    axi, zdi, zdstd, label, minner, maxer = tmp
    for node, idx in model.mapping.items():
        a = zdi[:, idx]  + sidx * zdstd[:, idx]
        b = zdi[:, idx]  - sidx * zdstd[:, idx]
        axi.plot(x, zdi[:, idx], linestyle = '--', \
                 markeredgecolor = 'black', label = node,\
                 color = colors[idx])
        axi.fill_between(x, a, b, color = colors[idx], alpha  = .4)
#        axi.yaxis.set_major_formatter(FormatStrFormatter('%1.0e'))
        
    axi.set(yticks = (minner, maxer))
    axi.ticklabel_format(axis = 'y', style = 'sci', scilimits = (0, 4))
    axi.set_ylabel(label, labelpad = -30)
axi.legend()
# %% root based on data
p0             = ones((func.__code__.co_argcount - 1)); p0[0] = 0
fitParam['p0'] = p0
aucs       = zeros((\
                        NSAMPLES, COND,\
                        NODES), dtype = float32)

from scipy import interpolate
from time import sleep

COEFFS = zeros((COND, NSAMPLES, NODES, p0.size))
x = arange(deltas // 2 + 1)

newx = linspace(0, deltas // 2 + 1, 100)

for samplei, sample in enumerate(zd):
    for condi, s in enumerate(sample.T):
        coeffs, errors = plotz.fit(s, func, params = fitParam)
        COEFFS[condi, samplei, ...] = coeffs
        for nodei, c in enumerate(coeffs):
            tmpF = lambda x: func(x, *c) 
            auc, er =  integrate.quad(func, \
                                      0, inf, args = tuple(c))
            aucs[samplei, condi, nodei] = auc
                
          

# %%
def showResults(data):
    
    fig, ax = subplots(1,2, sharex = 'all')
    mainax  = fig.add_subplot(111, frameon = False, 
                              **dict(xticks =  [], yticks = []))
    timer = 1.
    for idx, (c, stds, mus) in enumerate(zip(colors, data.std(0,ddof = 1).T, data.mean(0).T)):
        for axi, stdev, mu in zip(ax, stds, mus):
            # plot means
            axi.plot(thetas, mu, '-', color = c, \
                     markeredgewidth = 10, markeredgecolor = 'black',\
                     alpha = .7)
            axi.fill_between(thetas, mu - timer * stdev, \
                             mu + timer * stdev, color = c,\
                             alpha = .6)
            axi.set_xscale('log')
            axi.set_xlim(thetas[0], thetas[-1])
    
    ylabels = 'MI IMPACT'.split()            
    mainax.set_xlabel(r'$\theta$', labelpad = 30)
    mainax.set(**dict(title = 'Main Results\n'))
    axi.legend(list(model.mapping.keys()), bbox_to_anchor = (1.0, 1))
    tight_layout(w_pad = 5)
    [axi.set_ylabel(label) for axi, label in zip(ax, ylabels)]

fig, ax = subplots()

for idx, (c, i) in enumerate(zip( colors, aucs.T)):
    ax.scatter(*i, color = c, label = model.rmapping[idx])
    ax.scatter(*median(i, 1), marker = 's', color = c, edgecolors = 'k')
    ax.scatter(*mean(i, 1), marker = '^', color = c, edgecolors = 'k')
ax.legend(loc = 'upper right', bbox_to_anchor = (1.15, 1))
#ax.set(yscale = 'log')
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
#fig, ax = subplots(2)
#for axi, d in zip(ax, maxim.T):
#    h = axi.imshow(\
#       d.T, aspect = 'auto', vmax = nanmax(d), vmin = nanmin(d))
#    s = d.T.shape
#    colorbar(h, ax = axi)
#    axi.grid(which = 'minor', color = 'k', linestyle = '-', linewidth = 2)


# %%


# %%
